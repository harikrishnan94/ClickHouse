#!/usr/bin/env bash
# Shared harness for the divergence measurements (m_A.sh, m_B.sh, m_D1.sh, m_D2.sh, m_D3.sh).
# Sourced by those scripts; running it directly does nothing useful.
#
# Conventions inherited from tmp/uhj_versions_bench/{job_perf,deep_metrics_norm,thread_sweep}.sh:
#   * the 16-vCPU / 32 GiB cgroup comes from cgroup_wrap.sh --print-cg;
#   * the server must put ITSELF into the cgroup before exec, otherwise it sees the host's 96 CPUs
#     and picks a 96-wide default max_threads;
#   * TCP port 19010, one shared data directory, so only one script may run at a time;
#   * `hash` and `parallel_hash` are measured with clickhouse-baseline (merge-base, no D14 timers),
#     `unified_hash` with clickhouse-uhj.
#
# Differences from those scripts: this one writes its own config.xml so that
# system.processors_profile_log exists (the pre-existing config.xml only declares query_log), and it
# owns a lock so two measurements cannot fight over the port.

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
SRVROOT="${OUTROOT}/server"
LOCKFILE="${OUTROOT}/.lock"

mkdir -p "${OUTROOT}" "${SRVROOT}"

CUR_ARM=""
CUR_BIN=""

## ---------------------------------------------------------------------------------------------
## Mutual exclusion. stop_server kills every process in the cgroup, so two concurrent measurements
## would destroy each other's server. Fail loudly instead of racing.
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

# The arm that can run a given join_algorithm. unified_hash exists only in the branch binary;
# hash and parallel_hash must come from the merge-base binary so that D14's added
# ProfileEventTimeIncrement timers in ConcurrentHashJoin do not bias the baseline arm.
algo_arm() {
    case "$1" in
        unified_hash) echo "uhj" ;;
        hash|parallel_hash|'hash,parallel_hash'|'direct,parallel_hash,hash') echo "baseline" ;;
        *) echo "unknown join_algorithm '$1'" >&2; return 1 ;;
    esac
}

write_server_conf() {   # write_server_conf <arm>
    local arm="$1" dir="${SRVROOT}/$1"
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
    # join_algorithm is passed per query instead of being baked in here, because D1 needs three
    # different algorithm lists from the same binary.
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
## Settings shared by every synthetic isolation measurement.
##
##   collect_hash_table_stats_during_joins=0  the statistics cache changes the plan on one arm only
##                                            and dominates everything (harness convention).
##   enable_join_runtime_filters=0            a runtime filter can prune the probe side entirely,
##                                            which swamps the build-side effects under study.
##   enable_join_fixed_hash_table_conversion=0  post-build conversion to PartitionedFixedHashMap is
##                                            divergence D4, out of scope here.
##   query_plan_join_swap_table=false         pin the right table as the build side so that "build
##                                            side" means what the SQL says on both arms.
## ---------------------------------------------------------------------------------------------
##   query_plan_convert_join_to_in=0        pinned rather than trusted: if it ever flipped to true
##                                          by default, a build-only probe query would stop being a
##                                          join at all.
COMMON_SETTINGS=(
    --collect_hash_table_stats_during_joins=0
    --enable_join_runtime_filters=0
    --enable_join_fixed_hash_table_conversion=0
    --query_plan_join_swap_table=false
    --query_plan_convert_join_to_in=0
    --max_memory_usage=0
)

# Per-query hardware counters. These are opened by the query's own threads, so unlike a server-wide
# `perf stat` window they attribute exactly, need no iteration counting, and land in
# system.query_log.ProfileEvents next to JoinBuildTableRowCount - which makes
# "retired instructions per build row" a single SQL expression. Six events, so no PMU multiplexing.
#
# They can come back as zeros if perf_event_paranoid forbids self-monitoring; every script that
# relies on them calls perfev_available first and falls back to the perf_window path.
PERFEV_SETTINGS=(
    --metrics_perf_events_enabled=1
    --metrics_perf_events_list=PerfInstructions,PerfCPUCycles,PerfCacheMisses,PerfBranchMisses,PerfStalledCyclesBackend,PerfDataTLBMisses
)

perfev_available() {
    local qid="perfev_probe_$$_${RANDOM}_$(date +%s%N)" v
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

# Appended to every q_* call. Scripts set this to PERFEV_SETTINGS once hardware counters are known
# to work, so that individual call sites stay readable.
MEASURE_SETTINGS=()

# Wall-clock cap per query, in seconds. 0 disables it. Used by the suite censuses, where a handful
# of TPC-DS queries would otherwise run for minutes and add nothing.
Q_TIMEOUT="${Q_TIMEOUT:-0}"
q_run() {   # q_run <client args...>
    if [ "${Q_TIMEOUT}" -gt 0 ] 2>/dev/null; then
        timeout "${Q_TIMEOUT}" env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" "$@"
    else
        env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" "$@"
    fi
}

# q_time <query_id> <db> <sql> [extra settings...] -> prints seconds, or "null" on error
q_time() {
    local qid="$1" db="$2" sql="$3"; shift 3
    local out
    out="$(q_run --database "${db}" --query_id "${qid}" --time --format=Null \
            "${COMMON_SETTINGS[@]}" ${MEASURE_SETTINGS[@]+"${MEASURE_SETTINGS[@]}"} "$@" \
            --query "${sql}" 2>&1 | tail -1)"
    [[ "${out}" =~ ^[0-9]+\.[0-9]+$ ]] || out="null"
    echo "${out}"
}

# q_best <n> <query_id_prefix> <db> <sql> [extra settings...] -> prints "<min> <all times>"
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

# q_warm <db> <sql> [extra settings...] - run once, ignore the result (page cache / first-touch)
q_warm() {
    local db="$1" sql="$2"; shift 2
    q_run --database "${db}" --format=Null "${COMMON_SETTINGS[@]}" "$@" --query "${sql}" >/dev/null 2>&1 || true
}

# q_err <db> <sql> [extra settings...] - run and print the server's error text (one line), or "ok"
q_err() {
    local db="$1" sql="$2"; shift 2
    local out
    out="$(env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
            --database "${db}" --format=Null "${COMMON_SETTINGS[@]}" "$@" --query "${sql}" 2>&1 >/dev/null)" || true
    if [ -z "${out}" ]; then echo "ok"; else echo "${out}" | tr '\n' ' ' | cut -c1-400; fi
}

# q_ok <db> <sql> [extra settings...] - exit status only: 0 if the query succeeded
q_ok() {
    local db="$1" sql="$2"; shift 2
    env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --database "${db}" --format=Null "${COMMON_SETTINGS[@]}" "$@" --query "${sql}" >/dev/null 2>&1
}

# q_algorithm <db> <sql> [extra settings...] -> the join algorithm(s) the planner picked
q_algorithm() {
    local db="$1" sql="$2"; shift 2
    env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --database "${db}" "${COMMON_SETTINGS[@]}" "$@" \
        --query "EXPLAIN actions=1 ${sql}" 2>/dev/null \
        | sed -n 's/^ *Algorithm: *//p' | paste -sd, - | sed 's/^$/none/'
}

# q_build_streams <db> <sql> [extra settings...] -> number of FillingRightJoinSide processors
q_build_streams() {
    local db="$1" sql="$2"; shift 2
    env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --database "${db}" "${COMMON_SETTINGS[@]}" "$@" \
        --query "EXPLAIN PIPELINE ${sql}" 2>/dev/null | grep -c 'FillingRightJoinSide' || true
}

# q_maptype <db> <sql> [extra settings...] -> the "datatype:" the join chose (LOG_TEST line)
q_maptype() {
    local db="$1" sql="$2"; shift 2
    env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --database "${db}" --format=Null --send_logs_level=test "${COMMON_SETTINGS[@]}" "$@" \
        --query "${sql}" 2>&1 | sed -n 's/.*datatype: \([a-z0-9_]*\).*/\1/p' | sort -u | paste -sd, - | sed 's/^$/unknown/'
}

## ---------------------------------------------------------------------------------------------
## perf
## ---------------------------------------------------------------------------------------------
# Two 6-event groups so nothing multiplexes. cpu_cycles and inst_retired appear in both so the
# passes can be cross-checked against each other.
PERF_CORE='{cpu_cycles,inst_retired,stall_backend,stall_backend_mem,br_mis_pred_retired,mem_access}'
PERF_MEM='{cpu_cycles,inst_retired,l1d_cache_refill,ll_cache_miss_rd,dtlb_walk,mem_access}'
PERF_LOCK='syscalls:sys_enter_futex,syscalls:sys_enter_sched_yield'

perf_has_event() {   # perf_has_event <event>
    sudo perf list 2>/dev/null | tr -s ' ' '\n' | grep -qx -- "$1"
}

# perf_window <outfile-prefix> <events> <seconds> <loop-fn>
#   Runs <loop-fn> in a subshell until the window ends, counting completed iterations, while
#   perf stat samples the server process. Writes <prefix>.perf.csv and <prefix>.iters, and echoes
#   "iters=<n>".
perf_window() {
    local prefix="$1" events="$2" secs="$3" loopfn="$4"
    local cf="${prefix}.itercount" pid
    : > "${cf}"
    echo 0 > "${cf}"
    (
        n=0; end=$((SECONDS + secs + 8))
        while [ ${SECONDS} -lt ${end} ]; do
            "${loopfn}" && n=$((n+1))
            echo "${n}" > "${cf}"
        done
    ) &
    local loop=$!
    sleep 3                              # let the loop reach steady state before counting
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

# perf_value <prefix> <event> -> counter value, or 0.
# perf may report an event as "cpu_cycles" or as "armv8_pmuv3_0/cpu_cycles/", so match on substring.
perf_value() {
    awk -F, -v ev="$2" '
        index($3, ev) > 0 && !found { v = $1; gsub(/[^0-9]/, "", v); if (v != "") { print v; found = 1 } }
        END { if (!found) print 0 }' "$1.perf.csv" 2>/dev/null || echo 0
}

## ---------------------------------------------------------------------------------------------
## system.query_log / system.processors_profile_log extraction
## ---------------------------------------------------------------------------------------------
flush_logs() { client --query "SYSTEM FLUSH LOGS" >/dev/null 2>&1 || true; }

# Every dump below matches query ids with startsWith rather than LIKE: '_' is a single-character
# wildcard in LIKE, so 'mB_%' would also pick up the 'mBs_...' census ids.

# qlog_dump <query_id_prefix> <since_epoch> <outfile>
qlog_dump() {
    flush_logs
    client --query "
        SELECT query_id,
               round(query_duration_ms / 1000, 4)                            AS sec,
               memory_usage,
               ProfileEvents['JoinBuildTableRowCount']                       AS build_rows,
               ProfileEvents['JoinProbeTableRowCount']                       AS probe_rows,
               ProfileEvents['JoinResultRowCount']                           AS result_rows,
               ProfileEvents['PerfInstructions']                             AS instructions,
               ProfileEvents['PerfCPUCycles']                                AS cycles,
               ProfileEvents['PerfCacheMisses']                              AS cache_misses,
               ProfileEvents['PerfDataTLBMisses']                            AS dtlb_misses,
               ProfileEvents['PerfBranchMisses']                             AS branch_misses,
               ProfileEvents['PerfStalledCyclesBackend']                     AS stall_backend,
               if(build_rows > 0, round(instructions / build_rows, 2), 0)    AS instr_per_build_row,
               ProfileEvents['HashJoinPreallocatedElementsInHashTables']     AS prealloc,
               ProfileEvents['JoinSpillingHashJoinSwitchedToGraceJoin']      AS spilled,
               ProfileEvents['JoinBuildPostProcessingMicroseconds']          AS postbuild_us,
               ProfileEvents['JoinNonJoinedTransformRowCount']               AS nonjoined_rows
        FROM system.query_log
        WHERE type = 'QueryFinish'
          AND startsWith(query_id, '$1')
          AND event_time >= toDateTime($2)
        ORDER BY query_id
        FORMAT TSVWithNames" > "$3" 2>&1 || true
}

# qlog_agg <query_id_prefix> <since_epoch> <outfile>
#   One row per query_id prefix group (everything before the trailing _r<N> repeat marker), with the
#   minimum wall time and the median of each counter across repeats.
qlog_agg() {
    flush_logs
    client --query "
        SELECT replaceRegexpOne(query_id, '_r[0-9]+\$', '')                  AS tag,
               count()                                                       AS runs,
               min(query_duration_ms) / 1000                                 AS best_sec,
               round(median(query_duration_ms) / 1000, 4)                    AS med_sec,
               any(ProfileEvents['JoinBuildTableRowCount'])                  AS build_rows,
               any(ProfileEvents['JoinProbeTableRowCount'])                  AS probe_rows,
               toUInt64(median(ProfileEvents['PerfInstructions']))           AS instructions,
               toUInt64(median(ProfileEvents['PerfCPUCycles']))              AS cycles,
               toUInt64(median(ProfileEvents['PerfCacheMisses']))            AS cache_misses,
               toUInt64(median(ProfileEvents['PerfDataTLBMisses']))          AS dtlb_misses,
               toUInt64(median(ProfileEvents['PerfStalledCyclesBackend']))   AS stall_backend,
               if(build_rows > 0, round(instructions / build_rows, 2), 0)    AS instr_per_build_row,
               toUInt64(median(memory_usage))                                AS memory_usage
        FROM system.query_log
        WHERE type = 'QueryFinish'
          AND startsWith(query_id, '$1')
          AND event_time >= toDateTime($2)
        GROUP BY tag
        ORDER BY tag
        FORMAT TSVWithNames" > "$3" 2>&1 || true
}

# pplog_dump <query_id_prefix> <since_epoch> <outfile>
#   Per-processor totals. FillingRightJoinSide.elapsed_us summed over streams is the build-phase
#   CPU+lock time; JoiningTransform is the probe; NonJoinedBlocksTransform is the RIGHT/FULL tail.
pplog_dump() {
    flush_logs
    client --query "
        SELECT query_id, name,
               count()          AS streams,
               sum(elapsed_us)  AS elapsed_us,
               sum(input_wait_elapsed_us) AS input_wait_us,
               sum(input_rows)  AS input_rows
        FROM system.processors_profile_log
        WHERE startsWith(query_id, '$1')
          AND event_time >= toDateTime($2)
          AND name IN ('FillingRightJoinSide', 'JoiningTransform', 'NonJoinedBlocksTransform',
                       'DelayedJoinedBlocksWorkerTransform')
        GROUP BY query_id, name
        ORDER BY query_id, name
        FORMAT TSVWithNames" > "$3" 2>&1 || true
}

# qlog_field <query_id> <expression> -> one value from that query's query_log row, or 0
qlog_field() {
    flush_logs
    client --query "
        SELECT $2 FROM system.query_log
        WHERE query_id = '$1' AND type = 'QueryFinish'
        ORDER BY event_time_microseconds DESC LIMIT 1" 2>/dev/null || echo 0
}

# pp_build_us <query_id> -> summed FillingRightJoinSide elapsed_us for one query
pp_build_us() {
    flush_logs
    client --query "
        SELECT sum(elapsed_us) FROM system.processors_profile_log
        WHERE query_id = '$1' AND name = 'FillingRightJoinSide'" 2>/dev/null || echo 0
}

## ---------------------------------------------------------------------------------------------
## Synthetic fixtures. Idempotent: each table is created if missing and populated only when its row
## count does not match. Total footprint ~2.5 GB, well under the 20 GB budget.
##
## Key columns use CODEC(NONE) on purpose: LZ4 decompression would add a few hundred instructions
## per build row and dilute the per-row effects we are trying to see.
## ---------------------------------------------------------------------------------------------
SYNTH_DB=bench_synth

synth_rows() { client --query "SELECT count() FROM ${SYNTH_DB}.$1" 2>/dev/null || echo -1; }

# synth_table <name> <ddl-body> <select-for-insert> <expected-rows>
synth_table() {
    local name="$1" ddl="$2" sel="$3" want="$4" have
    client --query "CREATE TABLE IF NOT EXISTS ${SYNTH_DB}.${name} ${ddl}" >/dev/null
    have="$(synth_rows "${name}")"
    if [ "${have}" != "${want}" ]; then
        echo "#   populating ${SYNTH_DB}.${name} (have=${have}, want=${want})"
        client --query "TRUNCATE TABLE ${SYNTH_DB}.${name}" >/dev/null
        client --max_insert_threads=8 --max_memory_usage=0 \
               --query "INSERT INTO ${SYNTH_DB}.${name} ${sel}" >/dev/null
        client --query "OPTIMIZE TABLE ${SYNTH_DB}.${name} FINAL" >/dev/null 2>&1 || true
        have="$(synth_rows "${name}")"
        [ "${have}" = "${want}" ] || { echo "failed to populate ${name}: ${have} != ${want}" >&2; exit 1; }
    fi
}

ensure_synth() {
    echo "# ensuring ${SYNTH_DB} fixtures"
    client --query "CREATE DATABASE IF NOT EXISTS ${SYNTH_DB}" >/dev/null

    # --- build sides -----------------------------------------------------------------------
    # 64 M distinct UInt64 keys, PK-ordered so `WHERE id < N` is an exact, cheap cardinality knob.
    synth_table build_u64 \
        "(id UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY id" \
        "SELECT number FROM numbers_mt(64000000)" 64000000

    # 16 M distinct 48-byte String keys: the most expensive key getter that still uses one column,
    # and ineligible for the dense-keys fast path in scatterBlockBySlot.
    synth_table build_str48 \
        "(id UInt64 CODEC(NONE), k String CODEC(NONE)) ENGINE = MergeTree ORDER BY id" \
        "SELECT number, rightPad(hex(sipHash128(number)), 48, 'x') FROM numbers_mt(16000000)" 16000000

    # 16 M distinct 4-column UInt64 keys -> keys256 (32-byte packed key, UInt256HashCRC32).
    synth_table build_keys256 \
        "(id UInt64 CODEC(NONE), a UInt64 CODEC(NONE), b UInt64 CODEC(NONE), c UInt64 CODEC(NONE), d UInt64 CODEC(NONE))
         ENGINE = MergeTree ORDER BY id" \
        "SELECT number, number, number * 7 + 1, number * 13 + 2, number * 17 + 3 FROM numbers_mt(16000000)" 16000000

    # Nullable key with ~1% NULLs: on a RIGHT join this makes addBlockToJoin store a nullmap, which
    # is a second mandatory blocks_mutex acquisition per build block (D3).
    synth_table build_u64_null \
        "(id UInt64 CODEC(NONE), k Nullable(UInt64) CODEC(NONE)) ENGINE = MergeTree ORDER BY id" \
        "SELECT number, if(number % 100 = 0, NULL, number) FROM numbers_mt(16000000)" 16000000

    # UInt16 key -> the key16 fixed map: a 2^16-cell buffer allocated at construction that never
    # grows, so UHJ's insert-delta accounting can never see it (D2).
    synth_table build_u16 \
        "(k UInt16 CODEC(NONE)) ENGINE = MergeTree ORDER BY k" \
        "SELECT toUInt16(number) FROM numbers(1000)" 1000

    # --- probe sides -----------------------------------------------------------------------
    # One row: makes a query build-only, so per-build-row metrics are not diluted by the probe.
    synth_table probe_one \
        "(id UInt64) ENGINE = MergeTree ORDER BY id" \
        "SELECT 0" 1

    # 10 M probe rows whose keys are 0..999, so every probe row matches exactly one build row for
    # every dim_* table below and the join output size is constant across the sweep.
    synth_table probe_10m \
        "(k UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY k" \
        "SELECT number % 1000 FROM numbers_mt(10000000)" 10000000

    # --- D1: build sides straddling parallel_hash_join_threshold (default 100000) ------------
    # Separate physical tables so rhs_size_estimation is the table's own row count and the gate
    # decision is unambiguous.
    local n
    for n in 1000 10000 50000 99000 101000 200000 1000000; do
        synth_table "dim_${n}" \
            "(id UInt64 CODEC(NONE), pad UInt64 CODEC(NONE)) ENGINE = MergeTree ORDER BY id" \
            "SELECT number, number FROM numbers(${n})" "${n}"
    done

    # 100-row pair: the whole query is join setup, which is where an unnecessary 256-bucket map
    # plus N slots costs the most relative to the work done.
    synth_table tiny_dim \
        "(id UInt64) ENGINE = MergeTree ORDER BY id" \
        "SELECT number FROM numbers(100)" 100
    synth_table tiny_fact \
        "(k UInt64) ENGINE = MergeTree ORDER BY k" \
        "SELECT number FROM numbers(100)" 100

    echo "# fixtures ready"
}

## ---------------------------------------------------------------------------------------------
## Real-world suites
##
## The four loaded suites, one query per line, 1-based line numbers matching the .sql files.
## ---------------------------------------------------------------------------------------------
SUITES=(job tpch tpcds coffeeshop)

# census_explain <outfile> [extra settings...]
#   Plan-only census: for every query in every suite, record which join algorithm the planner
#   picked and how many FillingRightJoinSide streams the pipeline has. Costs nothing to run because
#   nothing is executed, and it is the primary detector for D1 (a query whose baseline plan says
#   `HashJoin` is one where unified_hash silently goes parallel instead).
census_explain() {
    local outfile="$1"; shift
    local suite qidx query algos streams
    printf 'suite\tq\talgorithms\tbuild_streams\n' > "${outfile}"
    for suite in "${SUITES[@]}"; do
        qidx=0
        while IFS= read -r query <&3; do
            [ -z "${query}" ] && continue
            query="${query%;}"
            qidx=$((qidx + 1))
            algos="$(q_algorithm "${suite}" "${query}" "$@")"
            streams="$(q_build_streams "${suite}" "${query}" "$@")"
            printf '%s\t%s\t%s\t%s\n' "${suite}" "${qidx}" "${algos}" "${streams}" >> "${outfile}"
        done 3< "${VB}/queries/${suite}.sql"
    done
}

# census_exec <query_id_prefix> [extra settings...]
#   Executes every suite query once so that per-query ProfileEvents (build rows, instructions) land
#   in system.query_log under `<prefix>_<suite>_q<idx>_r1`. Honours Q_TIMEOUT.
census_exec() {
    local prefix="$1"; shift
    local suite qidx query
    for suite in "${SUITES[@]}"; do
        qidx=0
        while IFS= read -r query <&3; do
            [ -z "${query}" ] && continue
            query="${query%;}"
            qidx=$((qidx + 1))
            q_time "${prefix}_${suite}_q${qidx}_r1" "${suite}" "${query}" "$@" >/dev/null || true
        done 3< "${VB}/queries/${suite}.sql"
        echo "#   census: ${suite} (${qidx} queries)"
    done
}

## ---------------------------------------------------------------------------------------------
## misc
## ---------------------------------------------------------------------------------------------
now_epoch() { date -u +%s; }

# want_arm <arm> - true when the caller asked for this arm (ARM unset means "all arms")
want_arm() { [ -z "${WANT_ARM:-}" ] || [ "${WANT_ARM}" = "$1" ]; }

# Short, underscore-free algorithm codes. Query ids are parsed with splitByChar('_', ...), so
# "parallel_hash" cannot appear in one.
algo_code() {
    case "$1" in
        hash)          echo h  ;;
        parallel_hash) echo ph ;;
        unified_hash)  echo uh ;;
        *)             echo "$1" | tr -d '_' ;;
    esac
}

# algo_flag <algorithm> - the client setting that pins the planner to it
algo_flag() { echo "--join_algorithm=$1"; }

hr() { printf '%s\n' "-------------------------------------------------------------------------------"; }

# tsv <file> <fields...>
tsv() { local f="$1"; shift; printf '%s\n' "$(printf '%s\t' "$@" | sed 's/\t$//')" >> "${f}"; }

# tsv_prune <file> <tag-prefix>
#   Every result file ends each row with the cell's `tag`. Dropping the rows of the cells we are
#   about to re-measure is what makes a second run of the same arm replace its numbers instead of
#   appending a second copy underneath them.
tsv_prune() {
    local f="$1" p="$2" tmp
    [ -s "${f}" ] || return 0
    tmp="${f}.tmp.$$"
    awk -F'\t' -v p="${p}" 'NR == 1 || index($NF, p) != 1' "${f}" > "${tmp}" && mv "${tmp}" "${f}"
}

# tsv_prune_field <file> <1-based column> <value> - the same thing for result files whose rows are
# identified by a column rather than by a tag (the perf files, one row per arm or algorithm).
tsv_prune_field() {
    local f="$1" c="$2" v="$3" tmp
    [ -s "${f}" ] || return 0
    tmp="${f}.tmp.$$"
    awk -F'\t' -v c="${c}" -v v="${v}" 'NR == 1 || $c != v' "${f}" > "${tmp}" && mv "${tmp}" "${f}"
}

# readable_to_bytes <string> - "1.25 MiB" -> 1310720. ClickHouse formats byte counts in exception
# messages with ReadableSize, so this is how a reported size is turned back into a number. Two
# decimal digits of precision, i.e. ~1% - fine for a sanity check, not for the headline number.
readable_to_bytes() {
    awk -v s="$1" 'BEGIN {
        if (match(s, /[0-9]+(\.[0-9]+)?[ ]*([KMGT]i)?B/) == 0) { print 0; exit }
        t = substr(s, RSTART, RLENGTH)
        v = t + 0
        m = 1
        if (index(t, "KiB")) m = 1024
        else if (index(t, "MiB")) m = 1024 * 1024
        else if (index(t, "GiB")) m = 1024 * 1024 * 1024
        else if (index(t, "TiB")) m = 1024 * 1024 * 1024 * 1024
        printf "%d\n", v * m
    }'
}
