#!/usr/bin/env bash
# Control experiment for the warm-state root cause.
#
# calculateHashTableCacheKeys() only assigns a HashTablesStatistics cache key when
# allowParallelHashJoin() is true, and that returns false unless `parallel_hash` is in
# join_algorithm. So the prediction is:
#
#   join_algorithm=parallel_hash,...  -> stats recorded on run 1, plan re-optimized on run 2+
#   join_algorithm=hash               -> no stats, plan identical on every run
#   join_algorithm=unified_hash       -> no stats, plan identical on every run  (same as `hash`)
#
# If baseline-with-`hash` behaves like unified_hash, the outlier is the stats path, not the
# join implementation. Runs on the BASELINE binary only, so the engine is held constant.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
OUT="${WORK}/control"
WRAP="${HERE}/cgroup_wrap.sh"
PORT=19010
BIN="${WORK}/bin/clickhouse-baseline"
SERVER_DIR="${WORK}/server_baseline"
mkdir -p "${OUT}" "${SERVER_DIR}/log"

cat > "${SERVER_DIR}/users.xml" <<'EOF'
<clickhouse>
    <profiles><default><max_memory_usage>0</max_memory_usage></default></profiles>
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
    nohup "${helper}" >"${OUT}/server.log" 2>&1 &
    for _ in $(seq 1 180); do server_alive && break; sleep 1; done
    server_alive || { echo "server did not start"; exit 1; }
}

case_run() {  # case_run <algo> <db> <name> <sqlfile> <runs>
    local algo="$1" db="$2" name="$3" sqlfile="$4" runs="$5"
    local q; q="$(cat "${sqlfile}")"
    local tag="${name}_${algo//,/+}"
    start_server   # fresh server => empty statistics cache
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1
    client --database "${db}" --join_algorithm="${algo}" --query "EXPLAIN PLAN actions=0 ${q}" > "${OUT}/${tag}.plan_cold.txt" 2>&1 || true
    echo "### ${name} join_algorithm=${algo}"
    for i in $(seq 1 "${runs}"); do
        local t
        t=$(client --database "${db}" --join_algorithm="${algo}" --query_id "ctl_${tag}_run${i}" --time --format=Null --query "${q}" 2>&1 | tail -1)
        echo "    run${i} ${t}s"
    done
    client --database "${db}" --join_algorithm="${algo}" --query "EXPLAIN PLAN actions=0 ${q}" > "${OUT}/${tag}.plan_warm.txt" 2>&1 || true
    diff -u "${OUT}/${tag}.plan_cold.txt" "${OUT}/${tag}.plan_warm.txt" > "${OUT}/${tag}.plan_diff.txt" 2>&1 || true
    if [ -s "${OUT}/${tag}.plan_diff.txt" ]; then
        echo "    PLAN: DIFFERS cold vs warm ($(wc -l < "${OUT}/${tag}.plan_diff.txt") lines)"
    else
        echo "    PLAN: identical cold vs warm"
    fi
    echo "    cold order: $(rg -o 'ReadFromMergeTree \([a-z_.]+\)' "${OUT}/${tag}.plan_cold.txt" | sed 's/ReadFromMergeTree //' | tr '\n' ' ')"
    echo "    warm order: $(rg -o 'ReadFromMergeTree \([a-z_.]+\)' "${OUT}/${tag}.plan_warm.txt" | sed 's/ReadFromMergeTree //' | tr '\n' ' ')"
    client --query "SYSTEM FLUSH LOGS" >/dev/null 2>&1 || true
    client --query "
        SELECT query_id, round(query_duration_ms/1000,3) AS sec,
               ProfileEvents['JoinBuildTableRowCount'] AS build_rows,
               ProfileEvents['RuntimeFilterRowsChecked'] AS rf_checked,
               ProfileEvents['RuntimeFilterBlocksProcessed'] AS rf_blocks
        FROM system.query_log
        WHERE type='QueryFinish' AND query_id LIKE 'ctl_${tag}_run%'
        ORDER BY event_time_microseconds FORMAT PrettyCompactMonoBlock
    " 2>&1 | sed 's/^/    /'
    stop_server
}

case_run "parallel_hash,hash" tpch  q8  "${WORK}/q_tpch8.sql"   2
case_run "hash"               tpch  q8  "${WORK}/q_tpch8.sql"   3
case_run "parallel_hash,hash" tpcds q54 "${WORK}/q_tpcds54.sql" 3
case_run "hash"               tpcds q54 "${WORK}/q_tpcds54.sql" 3
echo "CONTROL_DONE"
