#!/usr/bin/env bash
# Is the build-phase software prefetch actually paying off on each arm?
#
# Both build loops call map.prefetch() through the same helper, but it only fires when
# shouldUseJoinPrefetch() says the map is big enough. If uhj's per-slot maps fall under that
# threshold, uhj issues the same loads with no prefetch - which is exactly "same misses, less
# overlap". Toggling the setting separates the two cases:
#   baseline slows to uhj's level with prefetch off AND uhj does not move  -> uhj is not prefetching
#   both slow by the same factor                                          -> prefetch is not the cause
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"; WRAP="${HERE}/cgroup_wrap.sh"
VB="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}"
PORT=19010; QIDX="${QIDX:-64}"
ARM="${ARM:?}"
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
for mt in 16 1; do
  for pf in 1 0; do
    client --database job --format=Null "${EXTRA[@]}" --max_threads=$mt \
        --enable_software_prefetch_in_join=$pf --query "${Q}" >/dev/null 2>&1 || true
    out=""
    for i in 1 2 3; do
        t=$(client --database job --time --format=Null "${EXTRA[@]}" --max_threads=$mt \
              --enable_software_prefetch_in_join=$pf --query "${Q}" 2>&1 | tail -1)
        out="${out} ${t}"
    done
    echo "${ARM} max_threads=${mt} prefetch=${pf}:${out}"
  done
done
stop_server
echo "PFTEST_DONE ${ARM}"
