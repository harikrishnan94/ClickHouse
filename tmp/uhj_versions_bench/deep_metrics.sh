#!/usr/bin/env bash
# (b) + (c): where the cycles go inside RowRefList::insert, and the full counter set.
#
# Three passes per arm against a looping query, statistics off on both arms so the plan
# (and therefore the work) is identical:
#   pass 1  perf stat, core group   -> IPC, stall breakdown, branch misses
#   pass 2  perf stat, memory group -> mem_access, L1/L2/LL refills, dTLB walks
#   pass 3  perf record, cycles + memory events -> per-symbol attribution and annotation
# Event groups are kept to 6 counters so nothing is multiplexed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
OUT="${WORK}/deep"
WRAP="${HERE}/cgroup_wrap.sh"
VB="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}"
PORT=19010
SAMPLE="${SAMPLE:-30}"
mkdir -p "${OUT}"
ARM="${ARM:?}"; QIDX="${QIDX:-64}"
SYM='_ZN2DB10RowRefList6insertEmRNS_5ArenaE'
EXTRA=(--collect_hash_table_stats_during_joins=0)

case "${ARM}" in
    baseline) BIN="${WORK}/bin/clickhouse-baseline"; JOIN_XML="" ;;
    uhj)      BIN="${WORK}/bin/clickhouse-uhj";      JOIN_XML='<join_algorithm>unified_hash</join_algorithm>' ;;
esac
SERVER_DIR="${WORK}/server_${ARM}"; mkdir -p "${SERVER_DIR}/log"
cat > "${SERVER_DIR}/users.xml" <<EOF
<clickhouse>
    <profiles><default><max_memory_usage>0</max_memory_usage>${JOIN_XML}</default></profiles>
    <users><default><password></password><networks><ip>::/0</ip></networks>
        <profile>default</profile><quota>default</quota><access_management>1</access_management></default></users>
    <quotas><default><interval><duration>3600</duration></interval></default></quotas>
</clickhouse>
EOF
client() { env HOME=/tmp TZ=UTC "${BIN}" client --host 127.0.0.1 --port "${PORT}" "$@"; }
server_alive() { client --query 'SELECT 1' </dev/null >/dev/null 2>&1; }
stop_server() {
    for p in $(cat /sys/fs/cgroup/uhj_versions_bench/run/cgroup.procs 2>/dev/null); do
        case "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)" in
            *uhj_versions_bench*) kill "$p" 2>/dev/null || true ;;
        esac
    done
    for _ in $(seq 1 90); do server_alive || break; sleep 1; done
    fuser -k "${PORT}/tcp" 2>/dev/null || true
}
start_server() {
    stop_server
    local cg helper
    cg="$("${WRAP}" --print-cg | awk -F= '/^cg=/{print $2}')"
    helper="${SERVER_DIR}/start_in_cgroup.sh"
    printf '#!/bin/bash\necho $$ | sudo tee %s/cgroup.procs >/dev/null\nexec "%s" server --config-file="%s/config.xml"\n' \
        "${cg}" "${BIN}" "${SERVER_DIR}" > "${helper}"
    chmod +x "${helper}"
    nohup "${helper}" >/dev/null 2>&1 &
    for _ in $(seq 1 180); do server_alive && break; sleep 1; done
    server_alive || { echo "server did not start"; exit 1; }
}

Q="$(sed -n "${QIDX}p" "${VB}/queries/job.sql")"; Q="${Q%;}"
BASE="${OUT}/${ARM}_q${QIDX}"

loop_start() {
    ( end=$((SECONDS + SAMPLE + 8))
      while [ ${SECONDS} -lt ${end} ]; do
          client --database job --format=Null "${EXTRA[@]}" --query "${Q}" >/dev/null 2>&1 || true
      done ) &
    LOOP=$!
}
loop_stop() { kill "${LOOP}" 2>/dev/null || true; wait "${LOOP}" 2>/dev/null || true; }

start_server
for i in 1 2 3; do client --database job --time --format=Null "${EXTRA[@]}" --query "${Q}" >/dev/null 2>&1 || true; done
# Query count in the window lets every counter be normalised per query.
t=$(client --database job --time --format=Null "${EXTRA[@]}" --query "${Q}" 2>&1 | tail -1)
echo "${ARM} q${QIDX}: hot run ${t}s"
echo "${t}" > "${BASE}.hot_time.txt"
SPID="$(pgrep -f "${BIN} server" | head -1)"

echo "--- pass 1: core counters ---"
loop_start
sudo perf stat -p "${SPID}" \
  -e '{cpu_cycles,inst_retired,stall_frontend,stall_backend,stall_backend_mem,br_mis_pred_retired}' \
  -- sleep "${SAMPLE}" 2> "${BASE}.stat_core.txt" || true
loop_stop
cat "${BASE}.stat_core.txt"

echo "--- pass 2: memory counters ---"
loop_start
sudo perf stat -p "${SPID}" \
  -e '{cpu_cycles,mem_access,l1d_cache_refill,l2d_cache_refill,ll_cache_miss_rd,dtlb_walk}' \
  -- sleep "${SAMPLE}" 2> "${BASE}.stat_mem.txt" || true
loop_stop
cat "${BASE}.stat_mem.txt"

echo "--- pass 3: per-symbol attribution + annotation ---"
loop_start
sudo perf record -p "${SPID}" -o "${BASE}.deep.perf.data" \
  -e cpu_cycles -c 2000003 \
  -e l1d_cache_refill -c 100003 \
  -e ll_cache_miss_rd -c 20011 \
  -e dtlb_walk -c 10007 \
  -- sleep "${SAMPLE}" >/dev/null 2>&1 || true
loop_stop
sudo chown "$(id -u):$(id -g)" "${BASE}.deep.perf.data" 2>/dev/null || true

for ev in cpu_cycles l1d_cache_refill ll_cache_miss_rd dtlb_walk; do
    echo "### ${ev}: share of samples per symbol (top 6)"
    perf report -i "${BASE}.deep.perf.data" --stdio -g none --no-children --event "${ev}" \
        --percent-limit 1 2>/dev/null | rg '^\s+[0-9]+\.[0-9]+%' \
        | sed -E 's/<[^>]{30,}>/<...>/g' | cut -c1-120 | head -6
done | tee "${BASE}.per_symbol.txt"

echo "--- annotation of RowRefList::insert (cycles) ---"
perf annotate -i "${BASE}.deep.perf.data" --stdio --no-source --percent-limit 0.4 \
    --event cpu_cycles -s "${SYM}" 2>/dev/null | head -70 > "${BASE}.annotate.txt" || true
cat "${BASE}.annotate.txt"

stop_server
echo "DEEP_DONE ${ARM}"
