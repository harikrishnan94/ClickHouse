#!/usr/bin/env bash
# Mid-query perf profile: start the query, let it reach steady state, then sample.
# Sampling from t=0 would profile the build phase of a long probe-bound query.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
OUT="${WORK}/perf"
WRAP="${HERE}/cgroup_wrap.sh"
PORT=19010
mkdir -p "${OUT}"

ARM="${ARM:?ARM=baseline|uhj}"
DB="${DB:?DB=tpch|tpcds}"
NAME="${NAME:?NAME=q8|q54}"
SQLFILE="${SQLFILE:?path to sql}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"   # populate the stats cache first (the state under test)
SETTLE="${SETTLE:-20}"            # seconds to wait after query start before sampling
PERF_SECONDS="${PERF_SECONDS:-40}"

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
    nohup "${helper}" >"${OUT}/${ARM}_${NAME}.server.log" 2>&1 &
    for _ in $(seq 1 180); do server_alive && break; sleep 1; done
    server_alive || { echo "server did not start"; exit 1; }
}

Q="$(cat "${SQLFILE}")"
BASE="${OUT}/${ARM}_${NAME}"

start_server
echo "arm=${ARM} join_algorithm=$(client --query "SELECT getSetting('join_algorithm')")"

for i in $(seq 1 "${WARMUP_RUNS}"); do
    echo "warmup${i}: $(client --database "${DB}" --time --format=Null --query "${Q}" 2>&1 | tail -1)s"
done

echo "profiled run starting; sampling ${PERF_SECONDS}s after a ${SETTLE}s settle"
client --database "${DB}" --query_id "${ARM}_${NAME}_perf" --time --format=Null --query "${Q}" > "${BASE}.timed.txt" 2>&1 &
QPID=$!
sleep "${SETTLE}"
SPID="$(pgrep -f "${BIN} server" | head -1)"
sudo perf record -F 199 -g --call-graph fp -p "${SPID}" -o "${BASE}.perf.data" -- sleep "${PERF_SECONDS}" >/dev/null 2>&1 || true
sudo chown "$(id -u):$(id -g)" "${BASE}.perf.data" 2>/dev/null || true

# ClickHouse's own sampling profiler view of the same query, by symbol.
client --query "SYSTEM FLUSH LOGS" >/dev/null 2>&1 || true

wait "${QPID}" 2>/dev/null || true
echo "query time: $(cat "${BASE}.timed.txt")"

client --query "SYSTEM FLUSH LOGS" >/dev/null 2>&1 || true
perf report -i "${BASE}.perf.data" --stdio --no-children --percent-limit 0.3 2>/dev/null \
    | rg -v '^#|^$' | head -45 > "${BASE}.perf_report.txt" || true
echo "=== perf hotspots (${ARM} ${NAME}) ==="
cat "${BASE}.perf_report.txt"

client --query "
    SELECT
        round(query_duration_ms/1000,3) AS sec,
        formatReadableSize(memory_usage) AS mem,
        ProfileEvents['RuntimeFiltersCreated'] AS rf_created,
        ProfileEvents['RuntimeFilterRowsChecked'] AS rf_checked,
        ProfileEvents['RuntimeFilterRowsPassed'] AS rf_passed,
        ProfileEvents['HashJoinPreallocatedElementsInHashTables'] AS prealloc,
        ProfileEvents['SelectedRows'] AS sel_rows
    FROM system.query_log
    WHERE type='QueryFinish' AND query_id = '${ARM}_${NAME}_perf'
    FORMAT Vertical" > "${BASE}.events.txt" 2>&1 || true
cat "${BASE}.events.txt"
stop_server
echo "PERF_DONE ${ARM} ${NAME}"
