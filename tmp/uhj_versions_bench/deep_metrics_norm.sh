#!/usr/bin/env bash
# Same counters as deep_metrics.sh, but the loop counts its own iterations so every
# counter can be reported per query rather than per 30s window.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"; OUT="${WORK}/deep"; WRAP="${HERE}/cgroup_wrap.sh"
VB="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}"
PORT=19010; SAMPLE="${SAMPLE:-30}"; mkdir -p "${OUT}"
ARM="${ARM:?}"; QIDX="${QIDX:-64}"
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

run_pass() {  # run_pass <tag> <event-group>
    local tag="$1" events="$2" cnt=0
    local cf="${BASE}.${tag}.iters"
    : > "${cf}"
    ( n=0; end=$((SECONDS + SAMPLE + 8))
      while [ ${SECONDS} -lt ${end} ]; do
          client --database job --format=Null "${EXTRA[@]}" --query "${Q}" >/dev/null 2>&1 && n=$((n+1))
          echo "${n}" > "${cf}"
      done ) &
    local loop=$!
    sleep 2                      # let the loop reach steady state before counting
    cnt=$(cat "${cf}" 2>/dev/null || echo 0)
    sudo perf stat -p "$(pgrep -f "${BIN} server" | head -1)" -e "${events}" \
        -- sleep "${SAMPLE}" 2> "${BASE}.${tag}.txt" || true
    local cnt2; cnt2=$(cat "${cf}" 2>/dev/null || echo 0)
    kill "${loop}" 2>/dev/null || true; wait "${loop}" 2>/dev/null || true
    echo "ITERS_${tag}=$((cnt2 - cnt))"
    echo "$((cnt2 - cnt))" > "${BASE}.${tag}.itercount"
    cat "${BASE}.${tag}.txt"
}

start_server
for i in 1 2 3; do client --database job --format=Null "${EXTRA[@]}" --query "${Q}" >/dev/null 2>&1 || true; done
echo "=== ${ARM} q${QIDX} ==="
run_pass core '{cpu_cycles,inst_retired,stall_frontend,stall_backend,stall_backend_mem,br_mis_pred_retired}'
run_pass mem  '{cpu_cycles,mem_access,l1d_cache_refill,l2d_cache_refill,ll_cache_miss_rd,dtlb_walk}'
stop_server
echo "NORM_DONE ${ARM}"
