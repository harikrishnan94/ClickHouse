#!/usr/bin/env bash
# perf stat comparison for one JOB query, same plan on both arms.
# Both arms spend most of their time in the same function (RowRefList::insert), so the
# question is what makes uhj's execution of it more expensive: work, or memory behaviour.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
OUT="${WORK}/job_perf"
WRAP="${HERE}/cgroup_wrap.sh"
VB="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}"
PORT=19010
mkdir -p "${OUT}"
ARM="${ARM:?}"; QIDX="${QIDX:?}"; SAMPLE="${SAMPLE:-30}"
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
start_server
for i in 1 2 3; do client --database job --time --format=Null "${EXTRA[@]}" --query "${Q}" >/dev/null 2>&1 || true; done
n=0; tot=0
for i in 1 2 3 4 5; do
    t=$(client --database job --time --format=Null "${EXTRA[@]}" --query "${Q}" 2>&1 | tail -1)
    tot=$(awk -v a="$tot" -v b="$t" 'BEGIN{print a+b}'); n=$((n+1))
done
echo "${ARM} q${QIDX} mean of ${n} hot runs: $(awk -v t="$tot" -v n="$n" 'BEGIN{printf "%.3f", t/n}')s"

SPID="$(pgrep -f "${BIN} server" | head -1)"
( end=$((SECONDS + SAMPLE + 5))
  while [ ${SECONDS} -lt ${end} ]; do
      client --database job --format=Null "${EXTRA[@]}" --query "${Q}" >/dev/null 2>&1 || true
  done ) &
LOOP=$!
sudo perf stat -p "${SPID}" \
  -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend \
  -- sleep "${SAMPLE}" 2> "${OUT}/${ARM}_q${QIDX}.perfstat.txt" || true
kill "${LOOP}" 2>/dev/null || true; wait "${LOOP}" 2>/dev/null || true
echo "=== perf stat ${ARM} q${QIDX} ==="
cat "${OUT}/${ARM}_q${QIDX}.perfstat.txt"
stop_server
echo "PERFSTAT_DONE ${ARM} q${QIDX}"
