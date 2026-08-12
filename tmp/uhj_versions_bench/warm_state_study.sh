#!/usr/bin/env bash
# Decisive test of the warm-state hypothesis.
#
# The versions benchmark runs each query 6x against one long-lived server, so runs
# 2..6 see a hash-table-statistics cache populated by run 1. Two settings consume it:
#   collect_hash_table_stats_during_joins   -> map preallocation
#   join_runtime_filter_size_from_hash_table_stats -> runtime filter sizing
#
# For each arm x query this measures, on a FRESH server (empty stats cache):
#   cold  = run 1            (no stats yet)
#   warm  = runs 2..4        (stats from run 1)
#   nostats = 3 runs with collect_hash_table_stats_during_joins=0
#   norfstats = 3 runs with join_runtime_filter_size_from_hash_table_stats=0
# plus EXPLAIN before/after warm-up and runtime-filter ProfileEvents per run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
OUT="${WORK}/warmstate"
WRAP="${HERE}/cgroup_wrap.sh"
PORT=19010
mkdir -p "${OUT}"

ARM="${ARM:?ARM=baseline|uhj}"
case "${ARM}" in
    baseline) BIN="${WORK}/bin/clickhouse-baseline"; JOIN_XML="" ;;
    uhj)      BIN="${WORK}/bin/clickhouse-uhj";      JOIN_XML='<join_algorithm>unified_hash</join_algorithm>' ;;
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
    for _ in $(seq 1 90); do server_alive || break; sleep 1; done
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
    for _ in $(seq 1 180); do server_alive && break; sleep 1; done
    server_alive || { echo "server did not start"; tail -30 "${OUT}/${ARM}.server.log"; exit 1; }
    echo "server up join_algorithm=$(client --query "SELECT getSetting('join_algorithm')") max_threads=$(client --query "SELECT getSetting('max_threads')")"
}

timed() {  # timed <query_id> <db> <sql> [extra client args...]
    local qid="$1" db="$2" q="$3"; shift 3
    client --database "${db}" --query_id "${qid}" --time --format=Null "$@" --query "${q}" 2>&1 | tail -1
}

study() {
    local db="$1" name="$2" sqlfile="$3"
    local q; q="$(cat "${sqlfile}")"
    local base="${OUT}/${ARM}_${name}"

    # Fresh server per query => genuinely empty stats cache.
    start_server
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1

    client --database "${db}" --query "EXPLAIN PLAN actions=0 ${q}" > "${base}.plan_cold.txt" 2>&1 || true

    echo "--- ${ARM}/${name}: default settings, fresh stats cache ---"
    for i in 1 2 3; do
        echo "  run${i}(default) $(timed "${ARM}_${name}_def${i}" "${db}" "${q}")"
    done

    client --database "${db}" --query "EXPLAIN PLAN actions=0 ${q}" > "${base}.plan_warm.txt" 2>&1 || true
    # Write the full diff to a file; piping it through head would SIGPIPE under pipefail.
    diff -u "${base}.plan_cold.txt" "${base}.plan_warm.txt" > "${base}.plan_diff.txt" 2>&1 || true
    if [ -s "${base}.plan_diff.txt" ]; then
        echo "  PLAN: DIFFERS cold vs warm ($(wc -l < "${base}.plan_diff.txt") diff lines) -> ${base}.plan_diff.txt"
        echo "  cold join order: $(rg -o 'ReadFromMergeTree \([a-z_.]+\)' "${base}.plan_cold.txt" | tr '\n' ' ')"
        echo "  warm join order: $(rg -o 'ReadFromMergeTree \([a-z_.]+\)' "${base}.plan_warm.txt" | tr '\n' ' ')"
    else
        echo "  PLAN: identical cold vs warm"
    fi

    echo "--- ${ARM}/${name}: collect_hash_table_stats_during_joins=0 ---"
    for i in 1 2; do
        echo "  run${i}(nostats) $(timed "${ARM}_${name}_nostats${i}" "${db}" "${q}" --collect_hash_table_stats_during_joins=0)"
    done

    echo "--- ${ARM}/${name}: join_runtime_filter_size_from_hash_table_stats=0 ---"
    for i in 1 2; do
        echo "  run${i}(norfstats) $(timed "${ARM}_${name}_norfstats${i}" "${db}" "${q}" --join_runtime_filter_size_from_hash_table_stats=0)"
    done

    echo "--- ${ARM}/${name}: enable_join_runtime_filters=0 ---"
    for i in 1 2; do
        echo "  run${i}(norf) $(timed "${ARM}_${name}_norf${i}" "${db}" "${q}" --enable_join_runtime_filters=0)"
    done

    client --query "SYSTEM FLUSH LOGS" >/dev/null 2>&1 || true
    client --query "
        SELECT
            query_id,
            round(query_duration_ms/1000, 3) AS sec,
            formatReadableSize(memory_usage) AS mem,
            ProfileEvents['RuntimeFiltersCreated'] AS rf_created,
            ProfileEvents['RuntimeFilterRowsChecked'] AS rf_checked,
            ProfileEvents['RuntimeFilterRowsPassed'] AS rf_passed,
            ProfileEvents['RuntimeFilterBlocksSkipped'] AS rf_blk_skipped,
            ProfileEvents['HashJoinPreallocatedElementsInHashTables'] AS prealloc,
            ProfileEvents['SelectedRows'] AS sel_rows,
            ProfileEvents['SelectedMarks'] AS sel_marks
        FROM system.query_log
        WHERE type = 'QueryFinish' AND query_id LIKE '${ARM}_${name}_%'
        ORDER BY event_time_microseconds
        FORMAT PrettyCompactMonoBlock
    " > "${base}.events.txt" 2>&1 || true
    cat "${base}.events.txt"
    stop_server
}

study tpch  q8  "${WORK}/q_tpch8.sql"
study tpcds q54 "${WORK}/q_tpcds54.sql"
echo "WARMSTATE_DONE ${ARM}"
