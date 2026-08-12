#!/usr/bin/env bash
# Attribute retired instructions per symbol, both arms, same plan.
#
# The whole-query counters show uhj retiring ~1.02 G more instructions per execution of
# q64 than baseline. RowRefList::insert is byte-identical and called an identical number of
# times, so that billion is executed by the code AROUND it. This records inst_retired (not
# cycles) so the profile ranks by work done rather than by time waited.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"; OUT="${WORK}/instr"; WRAP="${HERE}/cgroup_wrap.sh"
VB="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}"
PORT=19010; SAMPLE="${SAMPLE:-30}"; QIDX="${QIDX:-64}"; mkdir -p "${OUT}"
ARM="${ARM:?}"
EXTRA=(--collect_hash_table_stats_during_joins=0)
[ -n "${MT:-}" ] && EXTRA+=(--max_threads=${MT})
TAG="${TAG:-}"
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
BASE="${OUT}/${ARM}${TAG}_q${QIDX}"
start_server
for i in 1 2 3; do client --database job --format=Null "${EXTRA[@]}" --query "${Q}" >/dev/null 2>&1 || true; done
CF="${BASE}.iters"; : > "${CF}"
( n=0; end=$((SECONDS + SAMPLE + 8))
  while [ ${SECONDS} -lt ${end} ]; do
      client --database job --format=Null "${EXTRA[@]}" --query "${Q}" >/dev/null 2>&1 && n=$((n+1))
      echo "${n}" > "${CF}"
  done ) &
LOOP=$!
sleep 2
c0=$(cat "${CF}" 2>/dev/null || echo 0)
sudo perf record -p "$(pgrep -f "${BIN} server" | head -1)" -o "${BASE}.instr.perf.data" \
    -e inst_retired -c 4000037 -- sleep "${SAMPLE}" >/dev/null 2>&1 || true
c1=$(cat "${CF}" 2>/dev/null || echo 0)
kill "${LOOP}" 2>/dev/null || true; wait "${LOOP}" 2>/dev/null || true
echo "ITERS=$((c1 - c0))" | tee "${BASE}.itercount"
sudo chown "$(id -u):$(id -g)" "${BASE}.instr.perf.data" 2>/dev/null || true
perf report -i "${BASE}.instr.perf.data" --stdio -g none --no-children --percent-limit 0.3 2>/dev/null \
  | rg '^\s+[0-9]+\.[0-9]+%' | sed -E 's/<[^>]{25,}>/<...>/g' | cut -c1-150 > "${BASE}.instr_by_symbol.txt" || true
echo "=== ${ARM}: retired instructions by symbol ==="
head -30 "${BASE}.instr_by_symbol.txt"
stop_server
echo "INSTR_DONE ${ARM}"
