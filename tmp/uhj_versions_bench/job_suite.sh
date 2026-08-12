#!/usr/bin/env bash
# Full JOB suite (113 queries) on one arm under one settings variant.
#
# JOB is fast enough (geomean ~0.08s) to re-run the whole suite per variant, which is what
# the +11% regression needs: the original number came from a single interleaved pair whose
# TPC-H/TPC-DS siblings turned out to be plan artifacts, so JOB has to be re-tested with the
# statistics path both on (as benchmarked) and off (apples-to-apples).
#
# Keeps the ClickBench versions contract: TRIES=6 (1 cold + 5 hot), drop_caches per query.
# Emits a TSV: qidx <tab> t1..t6  (null on error)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
OUT="${WORK}/job_study"
WRAP="${HERE}/cgroup_wrap.sh"
VB="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}"
PORT=19010
TRIES="${TRIES:-6}"
QUERY_TIMEOUT="${QUERY_TIMEOUT:-300}"
mkdir -p "${OUT}"

ARM="${ARM:?ARM=baseline|uhj}"
VARIANT="${VARIANT:-default}"     # default | nostats
case "${ARM}" in
    baseline) BIN="${WORK}/bin/clickhouse-baseline"; JOIN_XML="" ;;
    uhj)      BIN="${WORK}/bin/clickhouse-uhj";      JOIN_XML='<join_algorithm>unified_hash</join_algorithm>' ;;
esac
EXTRA=()
[ "${VARIANT}" = "nostats" ] && EXTRA+=(--collect_hash_table_stats_during_joins=0)

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
    nohup "${helper}" >"${OUT}/${ARM}_${VARIANT}.server.log" 2>&1 &
    for _ in $(seq 1 180); do server_alive && break; sleep 1; done
    server_alive || { echo "server did not start"; exit 1; }
}

TSV="${OUT}/job_${ARM}_${VARIANT}.tsv"
: > "${TSV}"

start_server
echo "arm=${ARM} variant=${VARIANT} join_algorithm=$(client --query "SELECT getSetting('join_algorithm')") max_threads=$(client --query "SELECT getSetting('max_threads')")"

qidx=0
while IFS= read -r query <&3; do
    [ -z "${query}" ] && continue
    query="${query%;}"
    qidx=$((qidx + 1))
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1
    row=""
    for i in $(seq 1 "${TRIES}"); do
        res=$(timeout "${QUERY_TIMEOUT}" env HOME=/tmp TZ=UTC "${BIN}" client --host 127.0.0.1 --port "${PORT}" \
                --database job --query_id "job_${ARM}_${VARIANT}_q${qidx}_r${i}" \
                --time --format=Null "${EXTRA[@]}" --query "${query}" 2>&1 | tail -1)
        [[ "${res}" =~ ^[0-9]+\.[0-9]+$ ]] || res="null"
        row+="${res}"
        [ "${i}" -ne "${TRIES}" ] && row+=$'\t'
    done
    printf '%s\t%s\n' "${qidx}" "${row}" >> "${TSV}"
    echo "q${qidx}: ${row}"
done 3< "${VB}/queries/job.sql"

client --query "SYSTEM FLUSH LOGS" >/dev/null 2>&1 || true
client --query "
    SELECT
        splitByChar('_', query_id)[4] AS q,
        splitByChar('_', query_id)[5] AS run,
        round(query_duration_ms/1000,4) AS sec,
        ProfileEvents['JoinBuildTableRowCount'] AS build_rows,
        ProfileEvents['RuntimeFiltersCreated'] AS rf_created,
        ProfileEvents['HashJoinPreallocatedElementsInHashTables'] AS prealloc,
        ProfileEvents['SelectedRows'] AS sel_rows
    FROM system.query_log
    WHERE type='QueryFinish' AND query_id LIKE 'job_${ARM}_${VARIANT}_q%'
    ORDER BY toUInt32(substring(q, 2)), run
    FORMAT TSVWithNames
" > "${OUT}/job_${ARM}_${VARIANT}.events.tsv" 2>&1 || true

stop_server
echo "JOB_STUDY_DONE ${ARM} ${VARIANT}"
