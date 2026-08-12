#!/usr/bin/env bash
# Profile one short query by running it back-to-back under perf for a fixed window.
# A single 0.5s JOB query is too short to sample, so the loop supplies the samples.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
OUT="${WORK}/job_perf"
WRAP="${HERE}/cgroup_wrap.sh"
VB="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}"
PORT=19010
mkdir -p "${OUT}"

ARM="${ARM:?ARM=baseline|uhj}"
QIDX="${QIDX:?QIDX=<1-based line in job.sql>}"
SECONDS_TO_SAMPLE="${SECONDS_TO_SAMPLE:-40}"
# Same-plan comparison: statistics off on both arms.
EXTRA=(--collect_hash_table_stats_during_joins=0)

case "${ARM}" in
    baseline) BIN="${WORK}/bin/clickhouse-baseline"; JOIN_XML="" ;;
    uhj)      BIN="${WORK}/bin/clickhouse-uhj";      JOIN_XML='<join_algorithm>unified_hash</join_algorithm>' ;;
esac
SERVER_DIR="${WORK}/server_${ARM}"
mkdir -p "${SERVER_DIR}/log"
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
    nohup "${helper}" >"${OUT}/${ARM}_q${QIDX}.server.log" 2>&1 &
    for _ in $(seq 1 180); do server_alive && break; sleep 1; done
    server_alive || { echo "server did not start"; exit 1; }
}

Q="$(sed -n "${QIDX}p" "${VB}/queries/job.sql")"
Q="${Q%;}"
BASE="${OUT}/${ARM}_q${QIDX}"

start_server
echo "arm=${ARM} q${QIDX} join_algorithm=$(client --query "SELECT getSetting('join_algorithm')")"
# Warm the page cache so we profile CPU, not I/O.
for i in 1 2 3; do
    echo "  warm${i}: $(client --database job --time --format=Null "${EXTRA[@]}" --query "${Q}" 2>&1 | tail -1)s"
done

SPID="$(pgrep -f "${BIN} server" | head -1)"
echo "sampling ${SECONDS_TO_SAMPLE}s while looping the query (server pid ${SPID})"
( end=$((SECONDS + SECONDS_TO_SAMPLE + 5))
  while [ ${SECONDS} -lt ${end} ]; do
      client --database job --format=Null "${EXTRA[@]}" --query "${Q}" >/dev/null 2>&1 || true
  done ) &
LOOP=$!
sudo perf record -F 299 -g --call-graph fp -p "${SPID}" -o "${BASE}.perf.data" -- sleep "${SECONDS_TO_SAMPLE}" >/dev/null 2>&1 || true
kill "${LOOP}" 2>/dev/null || true
wait "${LOOP}" 2>/dev/null || true
sudo chown "$(id -u):$(id -g)" "${BASE}.perf.data" 2>/dev/null || true

perf report -i "${BASE}.perf.data" --stdio -g none --no-children --percent-limit 0.8 2>/dev/null \
  | rg '^\s+[0-9]+\.[0-9]+%' | sed -E 's/<[^>]{40,}>/<...>/g' | cut -c1-170 | head -22 > "${BASE}.flat.txt" || true
echo "=== ${ARM} q${QIDX} hotspots ==="
cat "${BASE}.flat.txt"
stop_server
echo "JOBPERF_DONE ${ARM} q${QIDX}"
