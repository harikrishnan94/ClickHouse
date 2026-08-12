#!/usr/bin/env bash
# (a) Prove both arms call RowRefList::insert the same number of times with the same input.
#
# The function is instruction-identical in both binaries (same mangled symbol, shared
# RowRefs.h), so "same code" is settled statically. This counts the dynamic calls with a
# uprobe. Every hit traps, so the query runs far slower than normal - the count is the
# product here, not the timing.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
OUT="${WORK}/deep"
WRAP="${HERE}/cgroup_wrap.sh"
VB="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}"
PORT=19010
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

PROBE="rrl_${ARM}"
sudo perf probe -d "probe_clickhouse*:${PROBE}" >/dev/null 2>&1 || true
echo "adding uprobe on ${SYM} in ${BIN}"
sudo perf probe -x "${BIN}" --no-inlines --add "${PROBE}=${SYM}" 2>&1 | tail -3
PROBE_FULL="$(sudo perf probe -l 2>/dev/null | rg -o "probe_\S*:${PROBE}" | head -1)"
echo "probe = ${PROBE_FULL:-NONE}"
[ -z "${PROBE_FULL}" ] && { echo "FAILED to add probe"; exit 1; }

Q="$(sed -n "${QIDX}p" "${VB}/queries/job.sql")"; Q="${Q%;}"
start_server
# Warm the page cache first: the probe run should not also be paying for I/O.
client --database job --format=Null "${EXTRA[@]}" --query "${Q}" >/dev/null 2>&1 || true
SPID="$(pgrep -f "${BIN} server" | head -1)"

echo "counting calls for ONE execution (uprobe traps make this slow)..."
sudo perf stat -e "${PROBE_FULL}" -p "${SPID}" -- \
    env HOME=/tmp TZ=UTC "${BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --database job "${EXTRA[@]}" --format=Null --query "${Q}" \
    2> "${OUT}/${ARM}_q${QIDX}.callcount.txt" || true
echo "=== ${ARM} q${QIDX} RowRefList::insert call count ==="
cat "${OUT}/${ARM}_q${QIDX}.callcount.txt"

sudo perf probe -d "${PROBE_FULL}" >/dev/null 2>&1 || true
stop_server
echo "CALLCOUNT_DONE ${ARM}"
