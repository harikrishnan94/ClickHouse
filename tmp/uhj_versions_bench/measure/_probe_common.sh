#!/usr/bin/env bash
# Shared harness for the probe-path / behaviour divergence measurements
# (m_D5, m_D6, m_D7, m_D15, m_D17, m_D21, m_D22, m_D24).
#
# Sourced by those scripts; running it directly does nothing useful.
#
# Conventions inherited from tmp/uhj_versions_bench/{job_perf,deep_metrics_norm,thread_sweep}.sh and
# kept identical to the sibling measure/_common.sh so that results are comparable:
#   * the 16-vCPU / 32 GiB cgroup comes from cgroup_wrap.sh --print-cg;
#   * the server must put ITSELF into the cgroup before exec, otherwise it sees the host's 96 CPUs
#     and picks a 96-wide default max_threads;
#   * TCP port 19010, one shared data directory, so only one script may run at a time - this file
#     takes the same lock file as measure/_common.sh (${OUTROOT}/.lock), so the two families of
#     scripts exclude each other as well as themselves;
#   * `hash` and `parallel_hash` are measured with clickhouse-baseline (merge-base, so the D14
#     ProfileEventTimeIncrement timers added to ConcurrentHashJoin on this branch do not bias the
#     baseline arm), `unified_hash` with clickhouse-uhj.
#
# This file is deliberately self-contained: the sibling agents are still editing _common.sh and
# _maps_common.sh, so nothing here sources them.

set -euo pipefail

MEASURE_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_HOME="$(dirname "${MEASURE_HOME}")"
WORK="${BENCH_HOME}/work"                    # symlink to /mnt/data/uhj_versions_bench
WRAP="${BENCH_HOME}/cgroup_wrap.sh"
VB="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}"
PORT=19010

BIN_BASELINE="${WORK}/bin/clickhouse-baseline"
BIN_UHJ="${WORK}/bin/clickhouse-uhj"
OUTROOT="${WORK}/measure"
SRVROOT="${OUTROOT}/server_probe"
LOCKFILE="${OUTROOT}/.lock"

mkdir -p "${OUTROOT}" "${SRVROOT}"

CUR_ARM=""
CUR_BIN=""

## ---------------------------------------------------------------------------------------------
## Mutual exclusion. stop_server kills every process in the cgroup, so two concurrent
## measurements would destroy each other's server. Fail loudly instead of racing.
## ---------------------------------------------------------------------------------------------
m_take_lock() {
    exec 9>"${LOCKFILE}"
    if ! flock -n 9; then
        echo "another measurement already holds ${LOCKFILE} - refusing to start" >&2
        exit 1
    fi
    echo "$$" >&9
}

m_out_dir() {   # m_out_dir <id>
    local d="${OUTROOT}/$1"
    mkdir -p "${d}"
    echo "${d}"
}

## ---------------------------------------------------------------------------------------------
## Server lifecycle
## ---------------------------------------------------------------------------------------------
arm_bin() {
    case "$1" in
        baseline) echo "${BIN_BASELINE}" ;;
        uhj)      echo "${BIN_UHJ}" ;;
        *) echo "unknown arm '$1' (expected baseline|uhj)" >&2; return 1 ;;
    esac
}

# The arm that must run a given join_algorithm.
algo_arm() {
    case "$1" in
        unified_hash) echo "uhj" ;;
        hash|parallel_hash) echo "baseline" ;;
        *) echo "unknown join_algorithm '$1'" >&2; return 1 ;;
    esac
}

write_server_conf() {   # write_server_conf <arm>
    local dir="${SRVROOT}/$1"
    mkdir -p "${dir}/log"
    cat > "${dir}/config.xml" <<EOF
<clickhouse>
    <logger>
        <level>information</level>
        <log>${dir}/log/server.log</log>
        <errorlog>${dir}/log/server.err.log</errorlog>
        <size>200M</size>
        <count>2</count>
    </logger>
    <http_port>18110</http_port>
    <tcp_port>${PORT}</tcp_port>
    <path>${WORK}/server_shared/data/</path>
    <tmp_path>${WORK}/server_shared/tmp/</tmp_path>
    <user_files_path>${WORK}/server_shared/user_files/</user_files_path>
    <format_schema_path>${WORK}/server_shared/format_schemas/</format_schema_path>
    <access_control_path>${WORK}/server_shared/access/</access_control_path>
    <user_directories>
        <users_xml><path>${dir}/users.xml</path></users_xml>
        <local_directory><path>${WORK}/server_shared/access/</path></local_directory>
    </user_directories>
    <mark_cache_size>5368709120</mark_cache_size>
    <uncompressed_cache_size>0</uncompressed_cache_size>
    <mlock_executable>false</mlock_executable>
    <query_log>
        <database>system</database><table>query_log</table>
        <flush_interval_milliseconds>3000</flush_interval_milliseconds>
    </query_log>
    <processors_profile_log>
        <database>system</database><table>processors_profile_log</table>
        <flush_interval_milliseconds>3000</flush_interval_milliseconds>
    </processors_profile_log>
</clickhouse>
EOF
    cat > "${dir}/users.xml" <<EOF
<clickhouse>
    <profiles><default><max_memory_usage>0</max_memory_usage></default></profiles>
    <users><default><password></password><networks><ip>::/0</ip></networks>
        <profile>default</profile><quota>default</quota><access_management>1</access_management></default></users>
    <quotas><default><interval><duration>3600</duration></interval></default></quotas>
</clickhouse>
EOF
}

client() { env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" "$@"; }
server_alive() { [ -n "${CUR_BIN}" ] || return 1; client --query 'SELECT 1' </dev/null >/dev/null 2>&1; }

stop_server() {
    for p in $(cat /sys/fs/cgroup/uhj_versions_bench/run/cgroup.procs 2>/dev/null); do
        case "$(tr '\0' ' ' < "/proc/${p}/cmdline" 2>/dev/null)" in
            *uhj_versions_bench*) kill "${p}" 2>/dev/null || true ;;
        esac
    done
    for _ in $(seq 1 90); do server_alive || break; sleep 1; done
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 1
}

start_server() {   # start_server <arm>
    local arm="$1"
    CUR_ARM="${arm}"
    CUR_BIN="$(arm_bin "${arm}")"
    [ -x "${CUR_BIN}" ] || { echo "missing binary ${CUR_BIN}" >&2; exit 1; }
    stop_server
    write_server_conf "${arm}"
    local cg helper dir="${SRVROOT}/${arm}"
    cg="$("${WRAP}" --print-cg | awk -F= '/^cg=/{print $2}')"
    helper="${dir}/start_in_cgroup.sh"
    printf '#!/bin/bash\necho $$ | sudo tee %s/cgroup.procs >/dev/null\nexec "%s" server --config-file="%s/config.xml"\n' \
        "${cg}" "${CUR_BIN}" "${dir}" > "${helper}"
    chmod +x "${helper}"
    nohup "${helper}" >"${dir}/log/boot.log" 2>&1 &
    for _ in $(seq 1 180); do server_alive && break; sleep 1; done
    server_alive || { echo "server (${arm}) did not start; see ${dir}/log/" >&2; exit 1; }
    # A server that escaped the cgroup would see 96 CPUs and silently change every max_threads
    # default, so check rather than assume.
    local ncpu
    ncpu="$(client --query "SELECT value FROM system.settings WHERE name='max_threads'" 2>/dev/null || echo '?')"
    echo "# server up: arm=${arm} bin=$(basename "${CUR_BIN}") max_threads_default=${ncpu}"
}

server_pid() { pgrep -f "${CUR_BIN} server" | head -1; }

## ---------------------------------------------------------------------------------------------
## Settings shared by every cell.
##
##   collect_hash_table_stats_during_joins=0   the statistics cache changes the reserve on one arm
##                                             only and dominates everything (harness convention).
##   parallel_hash_join_threshold=0            removes D1 from every parallel_hash cell: without it
##                                             a build side under 100k rows silently runs serial
##                                             `hash` on the baseline arm while unified_hash goes
##                                             parallel. Ignored by `hash` and by unified_hash.
##   enable_join_runtime_filters=0             a runtime filter can prune the probe side (and for a
##                                             RIGHT join, change which right rows are visited),
##                                             which swamps everything measured here.
##   enable_join_fixed_hash_table_conversion=0 post-build conversion to PartitionedFixedHashMap is
##                                             divergence D4, out of scope for this group.
##   query_plan_join_swap_table=false          pin the right table as the build side so "build side"
##                                             means what the SQL says, on both arms.
##   query_plan_convert_join_to_in=0           pinned rather than trusted; m_D24.sh overrides it on
##                                             purpose, everything else needs the join to stay a join.
## ---------------------------------------------------------------------------------------------
COMMON_SETTINGS=(
    --collect_hash_table_stats_during_joins=0
    --parallel_hash_join_threshold=0
    --enable_join_runtime_filters=0
    --enable_join_fixed_hash_table_conversion=0
    --query_plan_join_swap_table=false
    --query_plan_convert_join_to_in=0
    --max_memory_usage=0
)

# Per-query hardware counters, opened by the query's own threads: they attribute exactly, need no
# iteration counting, and land in system.query_log.ProfileEvents next to JoinProbeTableRowCount,
# which makes "instructions per probe row" one SQL expression. Six events, so no PMU multiplexing.
# They come back as zeros if perf_event_paranoid forbids self-monitoring; call perfev_available
# first and fall back to wall time only.
PERFEV_SETTINGS=(
    --metrics_perf_events_enabled=1
    --metrics_perf_events_list=PerfInstructions,PerfCPUCycles,PerfCacheMisses,PerfBranchMisses,PerfStalledCyclesBackend,PerfDataTLBMisses
)

# Appended to every q_* call; scripts set it to PERFEV_SETTINGS once the counters are known to work.
MEASURE_SETTINGS=()

perfev_available() {
    local qid="pb_perfev_$$_${RANDOM}_$(date +%s%N)" v
    env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --query_id "${qid}" "${PERFEV_SETTINGS[@]}" --format=Null \
        --query "SELECT sum(number) FROM numbers(20000000)" >/dev/null 2>&1 || return 1
    flush_logs
    v="$(client --query "SELECT ProfileEvents['PerfInstructions'] FROM system.query_log
                         WHERE query_id = '${qid}' AND type = 'QueryFinish' LIMIT 1" 2>/dev/null || echo 0)"
    [ -n "${v}" ] && [ "${v}" != "0" ]
}

## ---------------------------------------------------------------------------------------------
## Query helpers
## ---------------------------------------------------------------------------------------------
Q_TIMEOUT="${Q_TIMEOUT:-0}"

q_run() {   # q_run <client args...>
    if [ "${Q_TIMEOUT}" -gt 0 ] 2>/dev/null; then
        timeout "${Q_TIMEOUT}" env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" "$@"
    else
        env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" "$@"
    fi
}

# q_time <query_id> <db> <sql> [extra settings...] -> seconds, or "null" on error
q_time() {
    local qid="$1" db="$2" sql="$3"; shift 3
    local out
    out="$(q_run --database "${db}" --query_id "${qid}" --time --format=Null \
            "${COMMON_SETTINGS[@]}" ${MEASURE_SETTINGS[@]+"${MEASURE_SETTINGS[@]}"} "$@" \
            --query "${sql}" 2>&1 | tail -1)"
    [[ "${out}" =~ ^[0-9]+\.[0-9]+$ ]] || out="null"
    echo "${out}"
}

# q_best <n> <query_id_prefix> <db> <sql> [extra settings...] -> "<min> <all times>"
q_best() {
    local n="$1" qid="$2" db="$3" sql="$4"; shift 4
    local i t all="" best=""
    for i in $(seq 1 "${n}"); do
        t="$(q_time "${qid}_r${i}" "${db}" "${sql}" "$@")"
        all="${all}${all:+ }${t}"
        if [[ "${t}" != "null" ]] && { [[ -z "${best}" ]] || awk -v a="${t}" -v b="${best}" 'BEGIN{exit !(a<b)}'; }; then
            best="${t}"
        fi
    done
    echo "${best:-null} ${all}"
}

# q_warm <db> <sql> [extra settings...] - run once, ignore the result (page cache / first touch)
q_warm() {
    local db="$1" sql="$2"; shift 2
    q_run --database "${db}" --format=Null "${COMMON_SETTINGS[@]}" "$@" --query "${sql}" >/dev/null 2>&1 || true
}

# q_scalar <db> <sql> [extra settings...] -> the single value the query returns, or "err"
q_scalar() {
    local db="$1" sql="$2"; shift 2
    q_run --database "${db}" "${COMMON_SETTINGS[@]}" "$@" --query "${sql}" 2>/dev/null | head -1 | tr -d '\r' \
        || echo err
}

# q_err <db> <sql> [extra settings...] -> "ok" or the server's error text on one line
q_err() {
    local db="$1" sql="$2"; shift 2
    local out
    out="$(env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
            --database "${db}" --format=Null "${COMMON_SETTINGS[@]}" "$@" --query "${sql}" 2>&1 >/dev/null)" || true
    if [ -z "${out}" ]; then echo "ok"; else echo "${out}" | tr '\n\t' '  ' | cut -c1-400; fi
}

# q_value_or_err <db> <sql> [extra settings...] -> "value=<v>" or "error=<text>"
# The user-visible answer for the D17 / D24 behaviour probes: it does not matter whether a query is
# slow, only whether it works and what it says when it does not.
q_value_or_err() {
    local db="$1" sql="$2"; shift 2
    local out rc=0
    out="$(env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
            --database "${db}" "${COMMON_SETTINGS[@]}" "$@" --query "${sql}" 2>&1)" || rc=$?
    out="$(echo "${out}" | tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-400)"
    if [ "${rc}" = 0 ]; then echo "value=${out}"; else echo "error=${out}"; fi
}

# q_algorithm <db> <sql> [extra settings...] -> the join algorithm(s) the planner picked
q_algorithm() {
    local db="$1" sql="$2"; shift 2
    env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --database "${db}" "${COMMON_SETTINGS[@]}" "$@" \
        --query "EXPLAIN actions=1 ${sql}" 2>/dev/null \
        | sed -n 's/^ *Algorithm: *//p' | sort -u | paste -sd, - | sed 's/^$/none/'
}

# q_pipeline_count <db> <processor-name> <sql> [extra settings...] -> how many such stages exist
q_pipeline_count() {
    local db="$1" name="$2" sql="$3"; shift 3
    env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --database "${db}" "${COMMON_SETTINGS[@]}" "$@" \
        --query "EXPLAIN PIPELINE ${sql}" 2>/dev/null | grep -c "${name}" || true
}

# q_plan_norm <db> <sql> [extra settings...] -> EXPLAIN with volatile bits removed, for diffing.
# Strips leading indentation, the __table<N> aliases the analyzer invents, and any hex/id noise, so
# that two plans differ only where the plan really differs.
q_plan_norm() {
    local db="$1" sql="$2"; shift 2
    env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --database "${db}" "${COMMON_SETTINGS[@]}" "$@" \
        --query "EXPLAIN actions=0, indexes=0 ${sql}" 2>&1 \
        | sed -e 's/^[[:space:]]*//' -e 's/__table[0-9]*/__tableN/g' -e 's/_[0-9a-f]\{8,\}/_HEX/g' \
        | grep -v '^$' || true
}

# q_maptype <db> <sql> [extra settings...] -> the "datatype:" the join chose (LOG_TEST line).
# This is how a cell proves which map layout it actually measured (key64 vs two_level_key64 ...).
q_maptype() {
    local db="$1" sql="$2"; shift 2
    env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --database "${db}" --format=Null --send_logs_level=test "${COMMON_SETTINGS[@]}" "$@" \
        --query "${sql}" 2>&1 | sed -n 's/.*datatype: \([a-z0-9_]*\).*/\1/p' | sort -u | paste -sd, - | sed 's/^$/unknown/'
}

## ---------------------------------------------------------------------------------------------
## perf (optional). Every caller must tolerate an empty result: sudo perf may be unavailable, and
## the per-query counters above are the primary source.
## ---------------------------------------------------------------------------------------------
PERF_MEM='{cpu_cycles,inst_retired,ll_cache_miss_rd,dtlb_walk,mem_access,br_mis_pred_retired}'

perf_usable() { sudo -n perf stat -e cpu_cycles -- true >/dev/null 2>&1; }

# perf_window <outfile-prefix> <events> <seconds> <loop-fn>
#   Runs <loop-fn> in a subshell until the window closes, counting completed iterations, while
#   perf stat samples the server process. Writes <prefix>.perf.csv and <prefix>.iters.
perf_window() {
    local prefix="$1" events="$2" secs="$3" loopfn="$4"
    local cf="${prefix}.itercount" pid
    echo 0 > "${cf}"
    (
        n=0; end=$((SECONDS + secs + 8))
        while [ ${SECONDS} -lt ${end} ]; do
            "${loopfn}" && n=$((n+1))
            echo "${n}" > "${cf}"
        done
    ) &
    local loop=$!
    sleep 3
    local c0 c1
    c0="$(cat "${cf}" 2>/dev/null || echo 0)"
    pid="$(server_pid)"
    if [ -n "${pid}" ]; then
        sudo perf stat -x, -p "${pid}" -e "${events}" -- sleep "${secs}" 2> "${prefix}.perf.csv" || true
    else
        echo "server pid not found" > "${prefix}.perf.csv"
        sleep "${secs}"
    fi
    c1="$(cat "${cf}" 2>/dev/null || echo 0)"
    kill "${loop}" 2>/dev/null || true
    wait "${loop}" 2>/dev/null || true
    echo "$((c1 - c0))" > "${prefix}.iters"
    echo "iters=$((c1 - c0))"
}

perf_value() {   # perf_value <prefix> <event>
    awk -F, -v ev="$2" '
        index($3, ev) > 0 && !found { v = $1; gsub(/[^0-9]/, "", v); if (v != "") { print v; found = 1 } }
        END { if (!found) print 0 }' "$1.perf.csv" 2>/dev/null || echo 0
}

## ---------------------------------------------------------------------------------------------
## system.query_log / system.processors_profile_log extraction
##
## Query ids are matched with startsWith, never LIKE: '_' is a single-character wildcard in LIKE,
## so 'pbD5_%' would also match 'pbD5x_...'.
## ---------------------------------------------------------------------------------------------
flush_logs() { client --query "SYSTEM FLUSH LOGS" >/dev/null 2>&1 || true; }

# qlog_agg <query_id_prefix> <since_epoch> <outfile>
#   One row per cell (query id minus the trailing _r<N> repeat marker): best wall time, median of
#   each counter, and the two per-row ratios that matter for this group.
qlog_agg() {
    flush_logs
    client --query "
        SELECT replaceRegexpOne(query_id, '_r[0-9]+\$', '')                  AS tag,
               count()                                                       AS runs,
               round(min(query_duration_ms) / 1000, 4)                       AS best_sec,
               round(median(query_duration_ms) / 1000, 4)                    AS med_sec,
               any(ProfileEvents['JoinBuildTableRowCount'])                  AS build_rows,
               any(ProfileEvents['JoinProbeTableRowCount'])                  AS probe_rows,
               any(ProfileEvents['JoinResultRowCount'])                      AS result_rows,
               any(ProfileEvents['JoinNonJoinedTransformRowCount'])          AS nonjoined_rows,
               toUInt64(median(ProfileEvents['PerfInstructions']))           AS instructions,
               toUInt64(median(ProfileEvents['PerfCPUCycles']))              AS cycles,
               toUInt64(median(ProfileEvents['PerfCacheMisses']))            AS cache_misses,
               toUInt64(median(ProfileEvents['PerfDataTLBMisses']))          AS dtlb_misses,
               toUInt64(median(ProfileEvents['PerfBranchMisses']))           AS branch_misses,
               toUInt64(median(ProfileEvents['PerfStalledCyclesBackend']))   AS stall_backend,
               if(probe_rows > 0, round(instructions / probe_rows, 3), 0)     AS instr_per_probe_row,
               if(build_rows > 0, round(instructions / build_rows, 3), 0)     AS instr_per_build_row,
               toUInt64(median(memory_usage))                                AS memory_usage
        FROM system.query_log
        WHERE type = 'QueryFinish'
          AND startsWith(query_id, '$1')
          AND event_time >= toDateTime($2)
        GROUP BY tag
        ORDER BY tag
        FORMAT TSVWithNames" > "$3" 2>&1 || true
}

# pplog_agg <query_id_prefix> <since_epoch> <outfile>
#   Per (cell, processor) totals. NonJoinedBlocksTransform is the RIGHT/FULL tail;
#   zero_out_streams counts the streams that produced no row at all, which is exactly what D21 is
#   about. Serial `hash` has no NonJoinedBlocksTransform: its non-joined rows are emitted from
#   inside JoiningTransform, so for that arm only the JoiningTransform row is meaningful.
pplog_agg() {
    flush_logs
    client --query "
        SELECT replaceRegexpOne(query_id, '_r[0-9]+\$', '') AS tag,
               name,
               count()                        AS streams,
               countIf(output_rows = 0)       AS zero_out_streams,
               sum(elapsed_us)                AS elapsed_us,
               max(elapsed_us)                AS max_stream_us,
               sum(output_rows)               AS output_rows,
               max(output_rows)               AS max_stream_rows
        FROM system.processors_profile_log
        WHERE startsWith(query_id, '$1')
          AND event_time >= toDateTime($2)
          AND name IN ('FillingRightJoinSide', 'JoiningTransform', 'NonJoinedBlocksTransform',
                       'DelayedJoinedBlocksWorkerTransform', 'DelayedJoinedBlocksTransform')
        GROUP BY tag, name
        ORDER BY tag, name
        FORMAT TSVWithNames" > "$3" 2>&1 || true
}

# pp_metric <query_id_prefix> <processor-name> <expression> -> one number over that cell's repeats
pp_metric() {
    flush_logs
    client --query "
        SELECT $3 FROM system.processors_profile_log
        WHERE startsWith(query_id, '$1') AND name = '$2'" 2>/dev/null | head -1 || echo 0
}

# qlog_metric <query_id_prefix> <expression> -> one number over that cell's repeats
qlog_metric() {
    flush_logs
    client --query "
        SELECT $2 FROM system.query_log
        WHERE type = 'QueryFinish' AND startsWith(query_id, '$1')" 2>/dev/null | head -1 || echo 0
}

## ---------------------------------------------------------------------------------------------
## Synthetic fixtures, all prefixed pb_ so they cannot collide with the sibling scripts' tables in
## the same database. Idempotent: a table is created when missing and repopulated only when its row
## count does not match. Key columns are CODEC(NONE) so that LZ4 decompression does not add a few
## hundred instructions per row on top of the per-row effects under study.
##
## Total footprint of all fixtures: ~3.3 GB, well inside the 20 GB budget. Each script asks only
## for the fixtures it needs, so a single-divergence run does not pay for all of them.
## ---------------------------------------------------------------------------------------------
SYNTH_DB=bench_synth

synth_rows() { client --query "SELECT count() FROM ${SYNTH_DB}.$1" 2>/dev/null || echo -1; }

# synth_table <name> <ddl-body> <select-for-insert> <expected-rows>
synth_table() {
    local name="$1" ddl="$2" sel="$3" want="$4" have
    client --query "CREATE DATABASE IF NOT EXISTS ${SYNTH_DB}" >/dev/null
    client --query "CREATE TABLE IF NOT EXISTS ${SYNTH_DB}.${name} ${ddl}" >/dev/null
    have="$(synth_rows "${name}")"
    if [ "${have}" != "${want}" ]; then
        echo "#   populating ${SYNTH_DB}.${name} (have=${have}, want=${want})"
        client --query "TRUNCATE TABLE ${SYNTH_DB}.${name}" >/dev/null
        if [ "${want}" != "0" ]; then
            client --max_insert_threads=8 --max_memory_usage=0 \
                   --query "INSERT INTO ${SYNTH_DB}.${name} ${sel}" >/dev/null
            client --query "OPTIMIZE TABLE ${SYNTH_DB}.${name} FINAL" >/dev/null 2>&1 || true
        fi
        have="$(synth_rows "${name}")"
        [ "${have}" = "${want}" ] || { echo "failed to populate ${name}: ${have} != ${want}" >&2; exit 1; }
    fi
}

# Number of build rows / probe rows in the big fixtures. Overridable so that a smoke run can be
# made cheap without editing the scripts: PB_SCALE=8 divides every big fixture by 8.
PB_SCALE="${PB_SCALE:-1}"
pb_n() { echo $(( $1 / PB_SCALE )); }

PB_DIM_ROWS="$(pb_n 32000000)"      # distinct UInt64 keys on the build side of the RIGHT/FULL shapes
PB_PROBE_ROWS="$(pb_n 32000000)"    # probe rows, key i -> build key i, so `WHERE k < X` sets the match rate
PB_K256_BUILD="$(pb_n 8000000)"     # distinct 4-column (32-byte) keys -> keys256, the dearest re-hash
PB_K256_PROBE="$(pb_n 48000000)"
PB_OR_BUILD="$(pb_n 8000000)"       # two-clause fixture; 8M cells is far larger than any L2
PB_OR_PROBE="$(pb_n 32000000)"
PB_ONEKEY_ROWS="$(pb_n 8000000)"    # one distinct key: the whole map lands in one routing bucket

# ensure_probe_synth <group...>, groups: right | k256 | orjoin | onekey | tiny
ensure_probe_synth() {
    local group
    for group in "$@"; do
        case "${group}" in
        right)
            # Build side of every RIGHT/FULL shape: PB_DIM_ROWS distinct UInt64 keys, one payload
            # column so the output has something to carry. The number of map cells - which is what
            # the non-joined scan walks - is the number of distinct keys.
            synth_table pb_dim_u64 \
                "(k UInt64 CODEC(NONE), pad UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY k" \
                "SELECT number, number FROM numbers_mt(${PB_DIM_ROWS})" "${PB_DIM_ROWS}"
            # Probe side, PK-ordered on the key, so `WHERE k < X` is an exact and cheap match-rate
            # knob: X/PB_DIM_ROWS of the right rows get matched, one probe row per right row.
            synth_table pb_probe_u64 \
                "(k UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY k" \
                "SELECT number FROM numbers_mt(${PB_PROBE_ROWS})" "${PB_PROBE_ROWS}"
            # Empty right side: the one window in which baseline's removed hasNonJoinedRows()
            # short-circuit actually fires (rows_to_join == 0).
            synth_table pb_dim_empty \
                "(k UInt64, pad UInt64) ENGINE = MergeTree ORDER BY k" \
                "SELECT 0, 0 WHERE 0" 0
            # UInt8 key: key8 has no two-level form, so parallel_hash keeps one FixedHashMap per
            # slot and takes the single-level branch of ConcurrentHashJoin::getNonJoinedBlocks -
            # the only branch that consults hasNonJoinedRows() at all.
            synth_table pb_dim_u8 \
                "(k UInt8 CODEC(NONE), pad UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY k" \
                "SELECT toUInt8(number % 256), number FROM numbers(256)" 256
            ;;
        k256)
            # Four UInt64 key columns pack into a 32-byte key -> keys256 with UInt256HashCRC32, the
            # most expensive re-hash of any join map, which is what baseline's offsetInternal pays
            # per matched probe row and per scanned cell.
            synth_table pb_k256_build \
                "(a UInt64 CODEC(NONE), b UInt64 CODEC(NONE), c UInt64 CODEC(NONE), d UInt64 CODEC(NONE),
                  pad UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY a" \
                "SELECT number, number * 7 + 1, number * 13 + 2, number * 17 + 3, number
                 FROM numbers_mt(${PB_K256_BUILD})" "${PB_K256_BUILD}"
            synth_table pb_k256_probe \
                "(i UInt64 CODEC(NONE), a UInt64 CODEC(NONE), b UInt64 CODEC(NONE), c UInt64 CODEC(NONE),
                  d UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY i" \
                "SELECT number,
                        number % ${PB_K256_BUILD},
                        (number % ${PB_K256_BUILD}) * 7 + 1,
                        (number % ${PB_K256_BUILD}) * 13 + 2,
                        (number % ${PB_K256_BUILD}) * 17 + 3
                 FROM numbers_mt(${PB_K256_PROBE})" "${PB_K256_PROBE}"
            # Single UInt64 key twin of the same shape, so "how much of the effect is the re-hash"
            # is answered by comparing key64 with keys256 at equal row counts.
            synth_table pb_u64_build \
                "(a UInt64 CODEC(NONE), pad UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY a" \
                "SELECT number, number FROM numbers_mt(${PB_K256_BUILD})" "${PB_K256_BUILD}"
            synth_table pb_u64_probe \
                "(i UInt64 CODEC(NONE), a UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY i" \
                "SELECT number, number % ${PB_K256_BUILD} FROM numbers_mt(${PB_K256_PROBE})" "${PB_K256_PROBE}"
            ;;
        orjoin)
            # Two-clause (OR) fixture. x and y live in disjoint ranges so a clause matches only
            # through its own column: ym probes the right side's y range and hits, yn probes a range
            # no right row occupies and misses. Both variants still touch both maps once per row.
            synth_table pb_or_r \
                "(x UInt64 CODEC(NONE), y UInt64 CODEC(NONE), pad UInt64 CODEC(NONE))
                 ENGINE = MergeTree ORDER BY x" \
                "SELECT number, number + 1000000000, number FROM numbers_mt(${PB_OR_BUILD})" "${PB_OR_BUILD}"
            synth_table pb_or_l \
                "(x UInt64 CODEC(NONE), ym UInt64 CODEC(NONE), yn UInt64 CODEC(NONE))
                 ENGINE = MergeTree ORDER BY x" \
                "SELECT number % ${PB_OR_BUILD},
                        (number % ${PB_OR_BUILD}) + 1000000000,
                        (number % ${PB_OR_BUILD}) + 2000000000
                 FROM numbers_mt(${PB_OR_PROBE})" "${PB_OR_PROBE}"
            ;;
        onekey)
            # One distinct key on the build side: every row routes to a single bucket, so exactly
            # one of the num_streams non-joined streams can ever produce a row. This is the
            # reachable upper bound on D21's stream imbalance.
            synth_table pb_onekey_r \
                "(k UInt64 CODEC(NONE), pad UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY k" \
                "SELECT 0, number FROM numbers_mt(${PB_ONEKEY_ROWS})" "${PB_ONEKEY_ROWS}"
            # Probe side that matches nothing, so every right row is non-joined and must be emitted.
            synth_table pb_nomatch_l \
                "(k UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY k" \
                "SELECT number + 1 FROM numbers(1000000)" 1000000
            ;;
        tiny)
            synth_table pb_small_l \
                "(k UInt64, v UInt64) ENGINE = MergeTree ORDER BY k" \
                "SELECT number, number FROM numbers(1000)" 1000
            synth_table pb_small_r \
                "(k UInt64, w UInt64) ENGINE = MergeTree ORDER BY k" \
                "SELECT number, number * 2 FROM numbers(1000)" 1000
            ;;
        *) echo "unknown fixture group '${group}'" >&2; exit 1 ;;
        esac
    done
}

## ---------------------------------------------------------------------------------------------
## Real-world exposure. The suites are one query per line, 1-based, matching the .sql files.
## suite_census needs no server: it is a textual classification of the query files, used to state
## how many suite queries can reach each divergence at all.
## ---------------------------------------------------------------------------------------------
SUITES=(job tpch tpcds coffeeshop)

# suite_census <outfile> - one row per (suite, query) that contains a join, with the flags that
# decide reachability for this divergence group.
suite_census() {
    local outfile="$1" suite
    printf 'suite\tq\thas_join\tright\tfull\tleft\tinner_kw\tcomma_join\ton_or\tany_semi_asof\twith_totals\tjoin_get\n' > "${outfile}"
    for suite in "${SUITES[@]}"; do
        awk -v s="${suite}" '
            BEGIN { IGNORECASE = 1 }
            {
                q = $0
                right = (q ~ /RIGHT[ ]+(OUTER[ ]+)?JOIN/) ? 1 : 0
                full  = (q ~ /FULL[ ]+(OUTER[ ]+)?JOIN/) ? 1 : 0
                left  = (q ~ /LEFT[ ]+(OUTER[ ]+)?JOIN/) ? 1 : 0
                inner = (q ~ /INNER[ ]+JOIN/) ? 1 : 0
                comma = (q ~ /FROM[ ]+[A-Za-z_][A-Za-z0-9_]*([ ]+AS[ ]+[A-Za-z_][A-Za-z0-9_]*)?[ ]*,[ ]*[A-Za-z_]/) ? 1 : 0
                joinkw = (q ~ /JOIN/) ? 1 : 0
                # An OR between two equalities inside an ON clause is what produces more than one
                # join clause; an OR of filter predicates does not. Flag the former only.
                on_or = (q ~ /ON[^;]*[A-Za-z0-9_.]+[ ]*=[ ]*[A-Za-z0-9_.]+[ ]*(\))?[ ]+OR[ ]+(\()?[ ]*[A-Za-z0-9_.]+[ ]*=[ ]*[A-Za-z0-9_.]+/) ? 1 : 0
                anysemi = (q ~ /(ANY|SEMI|ANTI|ASOF|PASTE)[ ]+(INNER[ ]+|LEFT[ ]+|RIGHT[ ]+|FULL[ ]+)?JOIN/) ? 1 : 0
                totals = (q ~ /WITH[ ]+TOTALS/) ? 1 : 0
                jget = (q ~ /joinGet/) ? 1 : 0
                hasjoin = (joinkw || comma) ? 1 : 0
                printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", s, NR, hasjoin, right, full, left, inner, comma, on_or, anysemi, totals, jget
            }' "${VB}/queries/${suite}.sql" >> "${outfile}"
    done
}

# suite_census_summary <census-file> - the counts quoted in SPEC_PROBE.md
suite_census_summary() {
    awk -F'\t' 'NR > 1 {
            n[$1]++
            for (i = 3; i <= 12; i++) c[$1, i] += $i
            tot++
            for (i = 3; i <= 12; i++) t[i] += $i
        }
        END {
            printf "%-11s %7s %7s %6s %5s %5s %8s %6s %6s %7s %8s\n",
                   "suite", "queries", "w/join", "RIGHT", "FULL", "LEFT", "INNERkw", "ON-OR", "ANY..", "TOTALS", "joinGet"
            for (s in n)
                printf "%-11s %7d %7d %6d %5d %5d %8d %6d %6d %7d %8d\n",
                       s, n[s], c[s,3], c[s,4], c[s,5], c[s,6], c[s,7], c[s,9], c[s,10], c[s,11], c[s,12]
            printf "%-11s %7d %7d %6d %5d %5d %8d %6d %6d %7d %8d\n",
                   "TOTAL", tot, t[3], t[4], t[5], t[6], t[7], t[9], t[10], t[11], t[12]
        }' "$1"
}

## ---------------------------------------------------------------------------------------------
## misc
## ---------------------------------------------------------------------------------------------
now_epoch() { date -u +%s; }

# want_arm <arm> - true when the caller asked for this arm (ARM unset means "all arms")
want_arm() { [ -z "${WANT_ARM:-}" ] || [ "${WANT_ARM}" = "$1" ]; }

# Short, underscore-free algorithm codes: query ids are split on '_', so "parallel_hash" must not
# appear inside one.
algo_code() {
    case "$1" in
        hash)          echo h  ;;
        parallel_hash) echo ph ;;
        unified_hash)  echo uh ;;
        *)             echo "$1" | tr -d '_' ;;
    esac
}

hr() { printf '%s\n' "-------------------------------------------------------------------------------"; }

# tsv <file> <fields...>
tsv() { local f="$1"; shift; printf '%s\n' "$(printf '%s\t' "$@" | sed 's/\t$//')" >> "${f}"; }

# tsv_head <file> <header fields...> - write the header only when the file is new, so re-runs append
tsv_head() {
    local f="$1"; shift
    [ -s "${f}" ] && return 0
    tsv "${f}" "$@"
}

# tsv_prune <file> <tag-prefix> - drop the rows of the cells about to be re-measured, which is what
# makes a second run replace its numbers instead of appending a second copy underneath them.
# Every result file in this group keeps the cell `tag` in column 1.
tsv_prune() {
    local f="$1" p="$2" tmp
    [ -s "${f}" ] || return 0
    tmp="${f}.tmp.$$"
    awk -F'\t' -v p="${p}" 'NR == 1 || index($1, p) != 1' "${f}" > "${tmp}" && mv "${tmp}" "${f}"
}

# done_sentinel <id> <outdir> - the last thing every script does
done_sentinel() {
    : > "$2/M_$1_DONE"
    echo "M_$1_DONE"
}
