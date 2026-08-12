#!/usr/bin/env bash
# Focused reproduction of the two outliers (tpch q8, tpcds q54) on both arms.
#
# Per arm and query, this records:
#   * 6 timed runs (1 cold + 5 hot) — the ClickBench versions contract
#   * per-run ProfileEvents (runtime filter, hash-table stats/preallocation)
#   * EXPLAIN PLAN + EXPLAIN PIPELINE
#   * a perf record of the server during one hot run -> hotspot report
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
OUT="${WORK}/repro"
WRAP="${HERE}/cgroup_wrap.sh"
SHARED_DIR="${WORK}/server_shared"
PORT=19010
PERF_SECONDS="${PERF_SECONDS:-25}"
mkdir -p "${OUT}"

ARM="${ARM:?ARM=baseline|uhj}"
case "${ARM}" in
    baseline) BIN="${WORK}/bin/clickhouse-baseline"; JOIN_XML="" ;;
    uhj)      BIN="${WORK}/bin/clickhouse-uhj";      JOIN_XML='<join_algorithm>unified_hash</join_algorithm>' ;;
    *) echo "bad ARM" >&2; exit 1 ;;
esac
SERVER_DIR="${WORK}/server_${ARM}"
mkdir -p "${SERVER_DIR}/log"

cat > "${SERVER_DIR}/users.xml" <<EOF
<clickhouse>
    <profiles><default>
        <max_memory_usage>0</max_memory_usage>
        ${JOIN_XML}
    </default></profiles>
    <users><default>
        <password></password><networks><ip>::/0</ip></networks>
        <profile>default</profile><quota>default</quota><access_management>1</access_management>
    </default></users>
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
    for _ in $(seq 1 60); do server_alive || break; sleep 1; done
    fuser -k "${PORT}/tcp" 2>/dev/null || true
}

start_server() {
    stop_server
    local cg helper
    cg="$("${WRAP}" --print-cg | awk -F= '/^cg=/{print $2}')"
    helper="${SERVER_DIR}/start_in_cgroup.sh"
    cat > "${helper}" <<EOF
#!/bin/bash
echo \$\$ | sudo tee ${cg}/cgroup.procs >/dev/null
exec "${BIN}" server --config-file="${SERVER_DIR}/config.xml"
EOF
    chmod +x "${helper}"
    nohup "${helper}" >"${OUT}/${ARM}.server.log" 2>&1 &
    echo $! > "${WORK}/repro_server.pid"
    for _ in $(seq 1 180); do server_alive && break; sleep 1; done
    server_alive || { echo "server did not start"; tail -30 "${OUT}/${ARM}.server.log"; exit 1; }
    SERVER_PID="$(pgrep -f "${BIN} server" | head -1)"
    echo "server up pid=${SERVER_PID} cgroup=$(cat /proc/${SERVER_PID}/cgroup) max_threads=$(client --query "SELECT getSetting('max_threads')") join_algorithm=$(client --query "SELECT getSetting('join_algorithm')")"
}

drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }

run_one() {
    local db="$1" name="$2" sqlfile="$3"
    local q; q="$(cat "${sqlfile}")"
    local base="${OUT}/${ARM}_${name}"

    echo "=== ${ARM} ${name}: EXPLAIN PLAN ==="
    client --database "${db}" --query "EXPLAIN PLAN actions=0, indexes=0 ${q}" > "${base}.explain_plan.txt" 2>&1 || true
    client --database "${db}" --query "EXPLAIN PIPELINE ${q}" > "${base}.explain_pipeline.txt" 2>&1 || true
    # Runtime-filter presence in the plan
    rg -c 'runtime|RuntimeFilter|__applyFilter' "${base}.explain_plan.txt" > "${base}.rf_plan_hits.txt" 2>&1 || echo 0 > "${base}.rf_plan_hits.txt"

    echo "=== ${ARM} ${name}: 6 timed runs (1 cold + 5 hot) ==="
    drop_caches
    : > "${base}.times.txt"
    for i in $(seq 1 6); do
        # Same timing contract as the ClickBench runner: client --time --format=Null.
        local qid="${ARM}_${name}_run${i}"
        local t
        t=$(client --database "${db}" --query_id "${qid}" --time --format=Null --query "${q}" 2>&1 | tail -1)
        echo "run${i} ${t} query_id=${qid}" | tee -a "${base}.times.txt"
    done

    client --query "SYSTEM FLUSH LOGS" >/dev/null 2>&1 || true
    echo "=== ${ARM} ${name}: per-run ProfileEvents ==="
    client --query "
        SELECT
            query_id,
            round(query_duration_ms / 1000, 3) AS sec,
            formatReadableSize(memory_usage) AS mem,
            arrayStringConcat(
                arrayMap(kv -> concat(kv.1, '=', toString(kv.2)),
                    arrayFilter(kv ->
                            position(kv.1, 'RuntimeFilter') > 0
                         OR position(kv.1, 'Preallocated') > 0
                         OR position(kv.1, 'HashTablesStat') > 0
                         OR position(kv.1, 'JoinBuild') > 0
                         OR kv.1 IN ('SelectedRows','SelectedMarks','SelectedParts'),
                        arrayZip(mapKeys(ProfileEvents), mapValues(ProfileEvents)))),
                ', ') AS events
        FROM system.query_log
        WHERE type = 'QueryFinish' AND query_id LIKE '${ARM}_${name}_run%'
        ORDER BY event_time_microseconds
        FORMAT Vertical
    " > "${base}.profile_events.txt" 2>&1 || true
    head -80 "${base}.profile_events.txt"
}

perf_one() {
    local db="$1" name="$2" sqlfile="$3"
    local q; q="$(cat "${sqlfile}")"
    local base="${OUT}/${ARM}_${name}"
    local spid; spid="$(pgrep -f "${BIN} server" | head -1)"
    echo "=== ${ARM} ${name}: perf record (pid ${spid}, ${PERF_SECONDS}s) ==="
    sudo perf record -F 199 -g --call-graph fp -p "${spid}" -o "${base}.perf.data" -- sleep "${PERF_SECONDS}" &
    local perfpid=$!
    client --database "${db}" --format=Null --query "${q}" >/dev/null 2>&1 || true
    wait "${perfpid}" 2>/dev/null || true
    sudo chown "$(id -u):$(id -g)" "${base}.perf.data" 2>/dev/null || true
    perf report -i "${base}.perf.data" --stdio --no-children --percent-limit 0.5 2>/dev/null \
        | head -60 > "${base}.perf_report.txt" || true
    echo "--- top hotspots ---"; head -35 "${base}.perf_report.txt"
}

# QUERIES selects which outliers to run; both scripts share one port and one cgroup,
# so never run two of these at once.
QUERIES="${QUERIES:-q8 q54}"
start_server
for sel in ${QUERIES}; do
    case "${sel}" in
        q8)  run_one tpch  q8  "${WORK}/q_tpch8.sql";   perf_one tpch  q8  "${WORK}/q_tpch8.sql" ;;
        q54) run_one tpcds q54 "${WORK}/q_tpcds54.sql"; perf_one tpcds q54 "${WORK}/q_tpcds54.sql" ;;
    esac
done
stop_server
echo "REPRO_DONE ${ARM}"
