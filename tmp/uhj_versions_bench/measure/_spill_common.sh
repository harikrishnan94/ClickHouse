#!/usr/bin/env bash
# Shared harness for the spilling / per-block-accounting / arena divergences:
#   m_D10 m_D11 m_D12 m_D14 m_D18 m_D19 m_D20 m_D23.
# Sourced by those scripts; running it directly does nothing.
#
# Deliberately self-contained rather than sourcing `_common.sh`: that file belongs to the
# high-impact and map-family measurements and is still being edited. The conventions are the same
# and the LOCK PATH IS THE SAME, so the three families exclude each other:
#   * the 16-vCPU / 32 GiB cgroup comes from cgroup_wrap.sh --print-cg;
#   * the server must put ITSELF into the cgroup before exec, or it sees the host's 96 CPUs and
#     picks a 96-wide default max_threads;
#   * TCP port 19010, one shared data directory, so only one script may run at a time;
#   * `hash` / `parallel_hash` / `grace_hash` normally come from clickhouse-baseline (the
#     merge-base build) and `unified_hash` from clickhouse-uhj.
#
# Two things are specific to this family.
#
# 1. SPILLING IS ON BY DEFAULT AND MUST BE TURNED OFF EXPLICITLY.
#    `max_bytes_before_external_join` defaults to 0, but `max_bytes_ratio_before_external_join`
#    defaults to 0.5 (`Core/Settings.cpp:8248`), and the effective threshold is the smaller of the
#    two non-zero values (`JoinOperator.cpp:301-334`). With a `tmp_path` configured, every eligible
#    hash join is therefore wrapped in `SpillingHashJoin` - the in-tree comment at
#    `Optimizations/topKThroughJoin.cpp:375` calls that "the steady state today". So every query
#    this harness issues passes BOTH settings explicitly: `sp_cap 0` means genuinely no wrapper,
#    `sp_cap N` means a wrapper with a threshold of exactly N bytes.
#
# 2. SOME CELLS RUN THE SAME ALGORITHM ON BOTH BINARIES.
#    D14 (the added `ConcurrentHashJoin` timers) and D20 (the added `shared_lock` on
#    `SpillingHashJoin`'s single-join path) are branch-vs-merge-base changes to the BASELINE arm.
#    Measuring them means running e.g. `parallel_hash` on clickhouse-baseline and on
#    clickhouse-uhj, so `sp_start` takes the binary, not the algorithm.
#
# No fixture tables. Every query sources from `numbers()` / `numbers_mt()`, which gives exactly
# `max_block_size` rows per block - so the block count, which is the independent variable for most
# of this group, is known exactly instead of being inferred from a MergeTree part layout. It also
# makes every script idempotent with nothing to create and nothing to clean up.

set -euo pipefail

SPILL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_HOME="$(dirname "${SPILL_HOME}")"
WORK="${BENCH_HOME}/work"                     # symlink to /mnt/data/uhj_versions_bench
WRAP="${BENCH_HOME}/cgroup_wrap.sh"
PORT="${PORT:-19010}"
HTTP_PORT="${HTTP_PORT:-18110}"

BIN_BASELINE="${WORK}/bin/clickhouse-baseline"
BIN_UHJ="${WORK}/bin/clickhouse-uhj"
OUTROOT="${WORK}/measure"
SRVROOT="${OUTROOT}/server_spill"
LOCKFILE="${OUTROOT}/.lock"

mkdir -p "${OUTROOT}" "${SRVROOT}"

CUR_ARM=""
CUR_BIN=""
OUT=""
SP_ID=""
START_TS=0

## ---------------------------------------------------------------------------------------------
## Mutual exclusion and output directory
## ---------------------------------------------------------------------------------------------

# `sp_stop` kills every process in the cgroup, so two concurrent measurements would destroy each
# other's server. Same lock file as `_common.sh` / `_maps_common.sh`, on purpose.
sp_take_lock() {
    exec 9>"${LOCKFILE}"
    if ! flock -n 9; then
        echo "another measurement already holds ${LOCKFILE} - refusing to start" >&2
        exit 1
    fi
    echo "$$" >&9
}

sp_init() {   # sp_init <id>
    SP_ID="$1"
    OUT="${OUTROOT}/${SP_ID}"
    mkdir -p "${OUT}"
    START_TS="$(date -u +%s)"
    rm -f "${OUT}/M_${SP_ID}_DONE"
    echo "== m_${SP_ID} =="
    echo "# out=${OUT} baseline=$(basename "${BIN_BASELINE}") uhj=$(basename "${BIN_UHJ}")"
}

sp_done() {
    : > "${OUT}/M_${SP_ID}_DONE"
    echo "artifacts: ${OUT}"
    echo "M_${SP_ID}_DONE"
}

hr() { printf '%s\n' '--------------------------------------------------------------------------------'; }

# want_arm <arm> - true when the caller asked for this arm (ARM unset means "every arm")
want_arm() { [ -z "${WANT_ARM:-}" ] || [ "${WANT_ARM}" = "$1" ]; }

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

# The arm that can run a given join_algorithm. `unified_hash` exists only in the branch binary.
# Everything else exists in both; the caller decides which one it wants, because for D14 and D20
# the point is to run the SAME algorithm on both.
algo_arm() {
    case "$1" in
        unified_hash) echo uhj ;;
        hash|parallel_hash|grace_hash) echo baseline ;;
        *) echo "unknown join_algorithm '$1'" >&2; return 1 ;;
    esac
}

sp_write_conf() {   # sp_write_conf <arm>
    local dir="${SRVROOT}/$1"
    mkdir -p "${dir}/log"
    # `processors_profile_log` is what separates build-transform time from probe-transform time and
    # from the source, which is the primary metric for most of this family. The pre-existing
    # config.xml in the benchmark tree only declares query_log, hence a config of our own.
    cat > "${dir}/config.xml" <<EOF
<clickhouse>
    <logger>
        <level>information</level>
        <log>${dir}/log/server.log</log>
        <errorlog>${dir}/log/server.err.log</errorlog>
        <size>200M</size>
        <count>2</count>
    </logger>
    <http_port>${HTTP_PORT}</http_port>
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
server_pid() { pgrep -f "${CUR_BIN} server" | head -1; }

sp_stop() {
    local p
    for p in $(cat /sys/fs/cgroup/uhj_versions_bench/run/cgroup.procs 2>/dev/null); do
        case "$(tr '\0' ' ' < "/proc/${p}/cmdline" 2>/dev/null)" in
            *uhj_versions_bench*) kill "${p}" 2>/dev/null || true ;;
        esac
    done
    for _ in $(seq 1 90); do server_alive || break; sleep 1; done
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 1
}

sp_start() {   # sp_start <arm>
    local arm="$1" cg helper dir ncpu
    CUR_ARM="${arm}"
    CUR_BIN="$(arm_bin "${arm}")"
    [ -x "${CUR_BIN}" ] || { echo "missing binary ${CUR_BIN}" >&2; exit 1; }
    sp_stop
    sp_write_conf "${arm}"
    dir="${SRVROOT}/${arm}"
    cg="$("${WRAP}" --print-cg | awk -F= '/^cg=/{print $2}')"
    helper="${dir}/start_in_cgroup.sh"
    printf '#!/bin/bash\necho $$ | sudo tee %s/cgroup.procs >/dev/null\nexec "%s" server --config-file="%s/config.xml"\n' \
        "${cg}" "${CUR_BIN}" "${dir}" > "${helper}"
    chmod +x "${helper}"
    nohup "${helper}" >"${dir}/log/boot.log" 2>&1 &
    for _ in $(seq 1 180); do server_alive && break; sleep 1; done
    server_alive || { echo "server (${arm}) did not start; see ${dir}/log/" >&2; exit 1; }
    # A server that escaped the cgroup sees 96 CPUs and silently changes every max_threads default,
    # so check rather than assume.
    ncpu="$(client --query "SELECT value FROM system.settings WHERE name='max_threads'" 2>/dev/null || echo '?')"
    echo "# server up: arm=${arm} bin=$(basename "${CUR_BIN}") max_threads_default=${ncpu}"
}

## ---------------------------------------------------------------------------------------------
## Settings
##
## Every cell sets the five knobs below and calls `sp_settings`; `SP_COMMON` is then the complete
## setting list for the cell. Composing them into one array rather than appending flags per call
## site avoids passing the same setting twice, which is how a cell silently stops testing what it
## says it tests.
##
##   SP_ALGO     join_algorithm, always a SINGLE algorithm. A single-element list also disables the
##               `parallel_hash_join_threshold` gate (`PlannerJoins.cpp:1244`), whose first term is
##               `!isEnabledAlgorithm(HASH)` - so `parallel_hash` alone is always parallel and
##               `hash` alone is always serial, and divergence D1 cannot leak into these cells.
##   SP_THREADS  max_threads.
##   SP_BS       max_block_size. With a `numbers()` source this is exactly the block size.
##   SP_CAP      max_bytes_before_external_join; 0 disables the SpillingHashJoin wrapper (see the
##               note at the top of the file - the ratio is pinned to 0 alongside it).
##   SP_STATS    collect_hash_table_stats_during_joins. 0 everywhere except m_D11, whose subject is
##               the statistics path itself.
##
## The rest are pinned for the same reasons the sibling harnesses pin them:
##   enable_join_runtime_filters=0             a runtime filter can prune the probe side entirely.
##   enable_join_fixed_hash_table_conversion=0 post-build conversion to PartitionedFixedHashMap is
##                                             divergence D4, out of scope here.
##   query_plan_join_swap_table=false          "build side" means what the SQL says, on both arms.
##   query_plan_convert_join_to_in=0           or a build-only query stops being a join.
##   preferred_block_size_bytes=0              no adaptive block sizing behind max_block_size.
## ---------------------------------------------------------------------------------------------
SP_ALGO="hash"
SP_THREADS=16
SP_BS=65505
SP_CAP=0
SP_STATS=0
SP_COMMON=()

sp_settings() {
    SP_COMMON=(
        --join_algorithm="${SP_ALGO}"
        --max_threads="${SP_THREADS}"
        --max_block_size="${SP_BS}"
        --max_bytes_before_external_join="${SP_CAP}"
        --max_bytes_ratio_before_external_join=0
        --collect_hash_table_stats_during_joins="${SP_STATS}"
        --enable_join_runtime_filters=0
        --enable_join_fixed_hash_table_conversion=0
        --query_plan_join_swap_table=false
        --query_plan_convert_join_to_in=0
        --preferred_block_size_bytes=0
        --max_memory_usage=0
    )
}
sp_settings

sp_cell() {   # sp_cell <algo> <threads> <block_size> [cap] [stats]
    SP_ALGO="$1"; SP_THREADS="$2"; SP_BS="$3"; SP_CAP="${4:-0}"; SP_STATS="${5:-0}"
    sp_settings
}

# Per-query hardware counters, opened by the query's own threads, so they attribute exactly and
# land in system.query_log.ProfileEvents next to the join row counts. Six events, no multiplexing.
# They come back as zeros where perf_event_paranoid forbids self-monitoring, hence sp_perfev_ok.
PERFEV_SETTINGS=(
    --metrics_perf_events_enabled=1
    --metrics_perf_events_list=PerfInstructions,PerfCPUCycles,PerfCacheMisses,PerfBranchMisses,PerfStalledCyclesBackend,PerfDataTLBMisses
)
MEASURE_SETTINGS=()

sp_perfev_ok() {
    local qid="spprobe_$$_${RANDOM}" v
    env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --query_id "${qid}" "${PERFEV_SETTINGS[@]}" --format=Null \
        --query "SELECT sum(number) FROM numbers(20000000)" >/dev/null 2>&1 || return 1
    sp_flush
    v="$(client --query "SELECT ProfileEvents['PerfInstructions'] FROM system.query_log
                         WHERE query_id = '${qid}' AND type = 'QueryFinish' LIMIT 1" 2>/dev/null || echo 0)"
    [ -n "${v}" ] && [ "${v}" != "0" ]
}

sp_enable_counters() {
    if sp_perfev_ok; then
        MEASURE_SETTINGS=("${PERFEV_SETTINGS[@]}")
        echo "# per-query hardware counters: available"
    else
        MEASURE_SETTINGS=()
        echo "# per-query hardware counters: NOT available (wall time and processor time only)"
    fi
}

## ---------------------------------------------------------------------------------------------
## Query helpers. All of them take the SQL as the first argument and pass "${SP_COMMON[@]}"; extra
## per-call flags go after the SQL.
## ---------------------------------------------------------------------------------------------
Q_TIMEOUT="${Q_TIMEOUT:-0}"
SP_DB=""      # when set, every sp_run query runs with --database=${SP_DB} (the suite passes)

sp_run() {
    local -a db=()
    [ -n "${SP_DB}" ] && db=(--database="${SP_DB}")
    if [ "${Q_TIMEOUT}" -gt 0 ] 2>/dev/null; then
        timeout "${Q_TIMEOUT}" env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
            ${db[0]+"${db[@]}"} "$@"
    else
        env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
            ${db[0]+"${db[@]}"} "$@"
    fi
}

# sp_time <query_id> <sql> [extra...] -> seconds, or "null"
sp_time() {
    local qid="$1" sql="$2"; shift 2
    local out
    out="$(sp_run --query_id "${qid}" --time --format=Null "${SP_COMMON[@]}" \
            ${MEASURE_SETTINGS[@]+"${MEASURE_SETTINGS[@]}"} "$@" --query "${sql}" 2>&1 | tail -1)"
    [[ "${out}" =~ ^[0-9]+\.[0-9]+$ ]] || out="null"
    echo "${out}"
}

# sp_best <n> <tag> <sql> [extra...] -> "<min> <all times>"; query ids are <tag>_r<i>
sp_best() {
    local n="$1" tag="$2" sql="$3"; shift 3
    local i t all="" best=""
    for i in $(seq 1 "${n}"); do
        t="$(sp_time "${tag}_r${i}" "${sql}" "$@")"
        all="${all}${all:+ }${t}"
        if [ "${t}" != null ] && { [ -z "${best}" ] || awk -v a="${t}" -v b="${best}" 'BEGIN{exit !(a<b)}'; }; then
            best="${t}"
        fi
    done
    echo "${best:-null} ${all}"
}

# sp_all <n> <tag> <sql> [extra...] -> every time in order, no minimum. For m_D11, where the
# INTERESTING thing is that run 1 differs from runs 2+.
sp_all() {
    local n="$1" tag="$2" sql="$3"; shift 3
    local i out=""
    for i in $(seq 1 "${n}"); do
        out="${out}${out:+ }$(sp_time "${tag}_r${i}" "${sql}" "$@")"
    done
    echo "${out}"
}

sp_warm() {   # sp_warm <sql> [extra...]
    local sql="$1"; shift
    sp_run --format=Null "${SP_COMMON[@]}" "$@" --query "${sql}" >/dev/null 2>&1 || true
}

sp_err() {   # sp_err <sql> [extra...] -> one-line server error, or "ok"
    local sql="$1"; shift
    local out
    out="$(sp_run --format=Null "${SP_COMMON[@]}" "$@" --query "${sql}" 2>&1 >/dev/null)" || true
    if [ -z "${out}" ]; then echo ok; else echo "${out}" | tr '\n' ' ' | cut -c1-300; fi
}

# sp_algorithm <sql> [extra...] -> the join object the planner built, from EXPLAIN's "Algorithm"
# field, which is `IJoin::getName()` (`JoinStep.cpp:51`). This is how a cell proves which regime it
# is in: "HashJoin", "ConcurrentHashJoin", "UnifiedHashJoin", "GraceHashJoin",
# "SpillingHashJoin(HashJoin)", "SpillingHashJoin(ConcurrentHashJoin)",
# "SpillingHashJoin(UnifiedHashJoin)".
sp_algorithm() {
    local sql="$1"; shift
    sp_run "${SP_COMMON[@]}" "$@" --query "EXPLAIN actions=1 ${sql}" 2>/dev/null \
        | sed -n 's/^ *Algorithm: *//p' | paste -sd, - | sed 's/^$/none/'
}

# sp_streams <processor-name> <sql> [extra...] -> how many of that processor the pipeline has
sp_streams() {
    local name="$1" sql="$2"; shift 2
    sp_run "${SP_COMMON[@]}" "$@" --query "EXPLAIN PIPELINE ${sql}" 2>/dev/null | grep -c "${name}" || true
}

# sp_maptype <sql> [extra...] -> the map the join chose, from the LOG_TEST "datatype:" line. Tells
# a 1-bucket serial map (`key64`) from a 256-bucket parallel one (`two_level_key64`), which is what
# D18 is about.
sp_maptype() {
    local sql="$1"; shift
    sp_run --format=Null --send_logs_level=test "${SP_COMMON[@]}" "$@" --query "${sql}" 2>&1 \
        | sed -n 's/.*datatype: \([a-z0-9_]*\).*/\1/p' | sort -u | paste -sd, - | sed 's/^$/unknown/'
}

# sp_trace_grep <pattern> <sql> [extra...] -> matching server trace lines for one execution
sp_trace_grep() {
    local pat="$1" sql="$2"; shift 2
    sp_run --format=Null --send_logs_level=trace "${SP_COMMON[@]}" "$@" --query "${sql}" 2>&1 \
        | grep -- "${pat}" | head -5 || true
}

## ---------------------------------------------------------------------------------------------
## The effective spill threshold
##
## `max_bytes_ratio_before_external_join` resolves against `getMostStrictAvailableSystemMemory()`
## at plan time and is logged at TRACE by `JoinSettings::getMaxBytesBeforeExternalJoin`
## (`JoinOperator.cpp:322`). Reading it back is the only way to state what the DEFAULT-settings
## threshold actually is on this machine, which is what decides the real-world exposure of D12,
## D18, D19 and D20.
## ---------------------------------------------------------------------------------------------
sp_default_threshold_report() {   # sp_default_threshold_report <outfile>
    local sql="SELECT count() FROM numbers_mt(1000000) AS l
               INNER JOIN (SELECT number AS k FROM numbers(100000)) AS r ON l.number = r.k"
    local saved_cap="${SP_CAP}" line algo
    {
        echo "# Effective max_bytes_before_external_join under DEFAULT spill settings"
        echo "# (max_bytes_before_external_join=0, max_bytes_ratio_before_external_join=0.5)"
        # Deliberately not sp_settings: this one cell must run with the DEFAULTS, since the whole
        # question is what a user - or a benchmark suite that never touches these settings - gets.
        line="$(sp_run --format=Null --send_logs_level=trace \
                    --join_algorithm=hash --max_threads=16 \
                    --query "${sql}" 2>&1 | grep -- 'Adjusting memory limit before external join' | head -1 || true)"
        if [ -n "${line}" ]; then
            echo "TRACE: ${line}"
        else
            echo "TRACE: no 'Adjusting memory limit before external join' line - either the ratio"
            echo "       resolved to nothing (no system memory limit visible) or the log level hid it."
        fi
        algo="$(sp_run --join_algorithm=hash --max_threads=16 \
                    --query "EXPLAIN actions=1 ${sql}" 2>/dev/null \
                    | sed -n 's/^ *Algorithm: *//p' | paste -sd, - | sed 's/^$/none/')"
        echo "join object with default spill settings, join_algorithm=hash:          ${algo}"
        algo="$(sp_run --join_algorithm=parallel_hash --max_threads=16 \
                    --query "EXPLAIN actions=1 ${sql}" 2>/dev/null \
                    | sed -n 's/^ *Algorithm: *//p' | paste -sd, - | sed 's/^$/none/')"
        echo "join object with default spill settings, join_algorithm=parallel_hash: ${algo}"
        if [ "${CUR_ARM}" = uhj ]; then
            algo="$(sp_run --join_algorithm=unified_hash --max_threads=16 \
                        --query "EXPLAIN actions=1 ${sql}" 2>/dev/null \
                        | sed -n 's/^ *Algorithm: *//p' | paste -sd, - | sed 's/^$/none/')"
            echo "join object with default spill settings, join_algorithm=unified_hash:  ${algo}"
        fi
        echo
        echo "A 'SpillingHashJoin(...)' above means the wrapper is present with DEFAULT settings,"
        echo "i.e. D20's extra shared_lock and the per-block getTotalByteCount() are paid by every"
        echo "eligible join in the loaded suites, and D12 changes the build-stream count there too."
    } > "$1" 2>&1
    SP_CAP="${saved_cap}"
    sp_settings
}

## ---------------------------------------------------------------------------------------------
## Real-world exposure: the four loaded suites
##
## One query per line in the ClickBench-versions .sql files, so the 1-based line number is the
## query number. Each suite is its own database, hence SP_DB.
## ---------------------------------------------------------------------------------------------
SP_SUITES=(job tpch tpcds coffeeshop)
SP_QDIR="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}/queries"

sp_suites_available() {
    local s
    for s in "${SP_SUITES[@]}"; do
        [ -s "${SP_QDIR}/${s}.sql" ] || return 1
    done
    return 0
}

# sp_census_explain <outfile>
#   Plan-only census with the CURRENT SP_COMMON settings: the join object the planner builds and how
#   many build streams the pipeline gets, per suite query. Nothing is executed, so this is cheap
#   enough to run unconditionally - and it is the whole deliverable for D12, whose content is a
#   stream count rather than a duration.
sp_census_explain() {
    local outfile="$1" suite qidx query algos streams
    printf 'arm\talgo\tsuite\tq\tjoin_objects\tbuild_streams\n' > "${outfile}"
    for suite in "${SP_SUITES[@]}"; do
        [ -s "${SP_QDIR}/${suite}.sql" ] || continue
        SP_DB="${suite}"
        qidx=0
        while IFS= read -r query <&3; do
            [ -z "${query}" ] && continue
            query="${query%;}"
            qidx=$((qidx + 1))
            algos="$(sp_algorithm "${query}")"
            streams="$(sp_streams FillingRightJoinSide "${query}")"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${CUR_ARM}" "${SP_ALGO}" "${suite}" "${qidx}" "${algos}" "${streams}" >> "${outfile}"
        done 3< "${SP_QDIR}/${suite}.sql"
        echo "#   census(explain): ${suite} (${qidx} queries)"
    done
    SP_DB=""
}

# sp_census_exec <query_id_prefix>
#   Executes every suite query once with the current settings, so its ProfileEvents land in
#   system.query_log under `<prefix>_<suite>_q<idx>_r1` and sp_dump can pick them up.
sp_census_exec() {
    local prefix="$1" suite qidx query
    for suite in "${SP_SUITES[@]}"; do
        [ -s "${SP_QDIR}/${suite}.sql" ] || continue
        SP_DB="${suite}"
        qidx=0
        while IFS= read -r query <&3; do
            [ -z "${query}" ] && continue
            query="${query%;}"
            qidx=$((qidx + 1))
            sp_time "${prefix}_${suite}_q${qidx}_r1" "${query}" >/dev/null || true
        done 3< "${SP_QDIR}/${suite}.sql"
        echo "#   census(exec): ${suite} (${qidx} queries)"
    done
    SP_DB=""
}

# sp_census_maptype <outfile>
#   Executes every suite query with LOG_TEST on and records the map each join chose. This is the
#   check behind D10's real-world claim ("no suite joins on a 1- or 2-byte key"), and it is a claim
#   worth checking rather than asserting. Expensive: it executes the suites.
sp_census_maptype() {
    local outfile="$1" suite qidx query maps
    printf 'arm\talgo\tsuite\tq\tmap_types\n' > "${outfile}"
    for suite in "${SP_SUITES[@]}"; do
        [ -s "${SP_QDIR}/${suite}.sql" ] || continue
        SP_DB="${suite}"
        qidx=0
        while IFS= read -r query <&3; do
            [ -z "${query}" ] && continue
            query="${query%;}"
            qidx=$((qidx + 1))
            maps="$(sp_maptype "${query}")"
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "${CUR_ARM}" "${SP_ALGO}" "${suite}" "${qidx}" "${maps}" >> "${outfile}"
        done 3< "${SP_QDIR}/${suite}.sql"
        echo "#   census(maptype): ${suite} (${qidx} queries)"
    done
    SP_DB=""
}

# sp_spill_history <outfile>
#   Whether anything in this benchmark has EVER spilled, read out of the shared data directory's
#   accumulated system.query_log rather than by running the suites again. The data directory has
#   carried every campaign run on this machine, so this is a free, empirical answer to the
#   real-world-exposure question for D18 and D19, over far more queries than one census would touch.
sp_spill_history() {
    {
        echo "# Spill evidence across the WHOLE accumulated system.query_log in the shared data dir"
        echo "# (every run of every script on this machine, not just this one)."
        echo
        echo "-- queries that actually switched to GraceHashJoin --"
        client --query "
            SELECT count()                                                        AS queries,
                   min(event_date)                                                AS first_day,
                   max(event_date)                                                AS last_day
            FROM system.query_log
            WHERE type = 'QueryFinish'
              AND ProfileEvents['JoinSpillingHashJoinSwitchedToGraceJoin'] > 0
            FORMAT Vertical" 2>&1 || true
        echo
        echo "-- queries that wrote any external-join temporary data --"
        client --query "
            SELECT count()                                                        AS queries,
                   toUInt64(max(ProfileEvents['ExternalJoinUncompressedBytes']))  AS max_uncompressed_bytes,
                   toUInt64(max(ProfileEvents['ExternalJoinWritePart']))          AS max_parts
            FROM system.query_log
            WHERE type = 'QueryFinish'
              AND ProfileEvents['ExternalJoinUncompressedBytes'] > 0
            FORMAT Vertical" 2>&1 || true
        echo
        echo "-- how close anything ever came: the largest per-query memory_usage on record --"
        client --query "
            SELECT toUInt64(max(memory_usage))                                    AS max_memory_usage,
                   formatReadableSize(max(memory_usage))                          AS max_memory_readable,
                   count()                                                        AS queries
            FROM system.query_log
            WHERE type = 'QueryFinish' AND memory_usage > 0
            FORMAT Vertical" 2>&1 || true
        echo
        echo "-- the ten hungriest queries on record, for context --"
        client --query "
            SELECT formatReadableSize(memory_usage)                               AS mem,
                   round(query_duration_ms / 1000, 2)                             AS sec,
                   substring(replaceRegexpAll(query, '\\\\s+', ' '), 1, 110)      AS q
            FROM system.query_log
            WHERE type = 'QueryFinish'
            ORDER BY memory_usage DESC
            LIMIT 10
            FORMAT TSV" 2>&1 || true
    } > "$1"
}

## ---------------------------------------------------------------------------------------------
## system.query_log / system.processors_profile_log
##
## One flush and one dump per pass, not per query: SYSTEM FLUSH LOGS costs more than most of the
## cells here. Wall time is echoed live from `client --time`; the counters arrive at the end.
## ---------------------------------------------------------------------------------------------
sp_flush() { client --query "SYSTEM FLUSH LOGS" >/dev/null 2>&1 || true; }

# sp_dump <query_id_prefix> <outfile>
#   One row per tag (the query id minus its trailing _r<N>), joining query_log with the two
#   join-related processors. Note startsWith rather than LIKE: '_' is a single-character wildcard
#   in LIKE, so 'spD1_%' would also match 'spD14...'.
#
#   build_us / probe_us are the summed elapsed_us of FillingRightJoinSide / JoiningTransform. They
#   exclude the `numbers()` source and are insensitive to how the work was spread over threads,
#   which makes them the right unit for a per-block constant.
sp_dump() {
    sp_flush
    client --query "
        WITH pp AS (
            SELECT query_id,
                   sumIf(elapsed_us, name = 'FillingRightJoinSide')             AS build_us,
                   countIf(name = 'FillingRightJoinSide')                       AS build_streams,
                   sumIf(elapsed_us, name = 'JoiningTransform')                 AS probe_us,
                   countIf(name = 'JoiningTransform')                           AS probe_streams,
                   sumIf(elapsed_us, name = 'NonJoinedBlocksTransform')         AS nonjoined_us,
                   sumIf(elapsed_us, name = 'DelayedJoinedBlocksWorkerTransform') AS delayed_us
            FROM system.processors_profile_log
            WHERE startsWith(query_id, '$1') AND event_time >= toDateTime(${START_TS})
            GROUP BY query_id
        )
        SELECT replaceRegexpOne(q.query_id, '_r[0-9]+\$', '')                   AS tag,
               count()                                                          AS runs,
               round(min(q.query_duration_ms) / 1000, 4)                        AS best_sec,
               round(median(q.query_duration_ms) / 1000, 4)                     AS med_sec,
               toUInt64(median(pp.build_us))                                    AS build_us,
               toUInt64(median(pp.probe_us))                                    AS probe_us,
               toUInt64(median(pp.nonjoined_us))                                AS nonjoined_us,
               toUInt64(median(pp.delayed_us))                                  AS delayed_us,
               toUInt64(any(pp.build_streams))                                  AS build_streams,
               toUInt64(any(pp.probe_streams))                                  AS probe_streams,
               toUInt64(median(q.memory_usage))                                 AS mem_bytes,
               toUInt64(any(q.ProfileEvents['JoinBuildTableRowCount']))         AS build_rows,
               toUInt64(any(q.ProfileEvents['JoinProbeTableRowCount']))         AS probe_rows,
               toUInt64(any(q.ProfileEvents['JoinResultRowCount']))             AS result_rows,
               toUInt64(median(q.ProfileEvents['PerfInstructions']))            AS instructions,
               toUInt64(median(q.ProfileEvents['PerfCPUCycles']))               AS cycles,
               toUInt64(median(q.ProfileEvents['PerfCacheMisses']))             AS cache_misses,
               toUInt64(median(q.ProfileEvents['PerfDataTLBMisses']))           AS dtlb_misses,
               toUInt64(median(q.ProfileEvents['ConcurrentHashJoinBuildMicroseconds']))         AS chj_build_us,
               toUInt64(median(q.ProfileEvents['ConcurrentHashJoinBuildDispatchMicroseconds'])) AS chj_bdisp_us,
               toUInt64(median(q.ProfileEvents['ConcurrentHashJoinBuildInsertMicroseconds']))   AS chj_bins_us,
               toUInt64(median(q.ProfileEvents['ConcurrentHashJoinProbeMicroseconds']))         AS chj_probe_us,
               toUInt64(median(q.ProfileEvents['ConcurrentHashJoinProbeDispatchMicroseconds'])) AS chj_pdisp_us,
               toUInt64(median(q.ProfileEvents['ConcurrentHashJoinProbeLookupMicroseconds']))   AS chj_plook_us,
               toUInt64(any(q.ProfileEvents['HashJoinPreallocatedElementsInHashTables']))       AS prealloc,
               toUInt64(max(q.ProfileEvents['JoinSpillingHashJoinSwitchedToGraceJoin']))        AS spilled,
               toUInt64(median(q.ProfileEvents['ExternalJoinWritePart']))       AS ext_parts,
               toUInt64(median(q.ProfileEvents['ExternalJoinMerge']))           AS ext_merges,
               toUInt64(median(q.ProfileEvents['ExternalJoinCompressedBytes'])) AS ext_comp_bytes,
               toUInt64(median(q.ProfileEvents['ExternalJoinUncompressedBytes'])) AS ext_uncomp_bytes,
               toUInt64(median(q.ProfileEvents['JoinBuildPostProcessingMicroseconds'])) AS postbuild_us
        FROM system.query_log AS q
        LEFT JOIN pp ON pp.query_id = q.query_id
        WHERE q.type = 'QueryFinish'
          AND startsWith(q.query_id, '$1')
          AND q.event_time >= toDateTime(${START_TS})
        GROUP BY tag
        ORDER BY tag
        FORMAT TSVWithNames" > "$2" 2>&1 || true
}

## ---------------------------------------------------------------------------------------------
## Result files
##
## Every grid row ends with its `tag`, which is also the query-id prefix of its repeats. Pruning by
## tag before re-measuring a cell is what makes a second run of a script replace numbers instead of
## appending a second copy underneath them.
## ---------------------------------------------------------------------------------------------
sp_tsv() { local f="$1"; shift; printf '%s\n' "$(printf '%s\t' "$@" | sed 's/\t$//')" >> "${f}"; }

sp_tsv_head() {   # sp_tsv_head <file> <header...>
    local f="$1"; shift
    [ -s "${f}" ] || printf '%s\n' "$(printf '%s\t' "$@" | sed 's/\t$//')" > "${f}"
}

sp_tsv_prune() {   # sp_tsv_prune <file> <tag-prefix>
    local f="$1" p="$2" tmp
    [ -s "${f}" ] || return 0
    tmp="${f}.tmp.$$"
    awk -F'\t' -v p="${p}" 'NR == 1 || index($NF, p) != 1' "${f}" > "${tmp}" && mv "${tmp}" "${f}"
}

# sp_join <grid> <dump> <outfile>
#   Append every dump column to the grid row whose last column is the matching tag. The grid holds
#   the cell's axes (arm, algorithm, threads, block size, ...) and the dump holds the counters;
#   joining them gives one wide table per pass that the summary awk can read.
sp_join() {
    awk -F'\t' -v OFS='\t' '
        FNR == NR {
            if (FNR == 1) { dh = ""; for (i = 2; i <= NF; i++) dh = dh OFS $i; nd = NF - 1; next }
            row = ""; for (i = 2; i <= NF; i++) row = row OFS $i
            d[$1] = row
            next
        }
        FNR == 1 { print $0 dh; next }
        {
            if ($NF in d) { print $0 d[$NF] }
            else { pad = ""; for (i = 0; i < nd; i++) pad = pad OFS "0"; print $0 pad }
        }' "$2" "$1" > "$3"
}

## ---------------------------------------------------------------------------------------------
## The cost of one ProfileEventTimeIncrement
##
## `ProfileEventTimeIncrement<Microseconds>` (`Common/ElapsedTimeProfileEventIncrement.h:18`) is a
## `Stopwatch(CLOCK_MONOTONIC)` plus a `ProfileEvents::increment` in its destructor: two
## `clock_gettime(CLOCK_MONOTONIC)` calls (vDSO, no syscall) and one increment, which is a
## `sched_getcpu` (an rseq TLS read, `Common/PerCPU.h:26`) plus one relaxed `fetch_add` per level of
## the counter chain - thread, thread group, global.
##
## Measuring that directly with a C program gives D14's per-timer cost to a few nanoseconds, which
## is far more precise than differencing two noisy wall times, and it converts "how many blocks"
## into "how many microseconds" without any further assumptions. Compiled and run inside the same
## cgroup as the server, so the clock source and the CPU are the same.
## ---------------------------------------------------------------------------------------------
sp_timer_cost() {   # sp_timer_cost <outfile> -> prints ns per timer pair on stdout too
    local out="$1" src="${OUT}/timer_cost.c" bin="${OUT}/timer_cost" ns
    if ! command -v cc >/dev/null 2>&1; then
        echo "0" > "${out}"
        echo "# no C compiler: skipping the direct ProfileEventTimeIncrement cost calibration" >&2
        echo 0
        return 0
    fi
    cat > "${src}" <<'EOF'
/* Cost of one ProfileEventTimeIncrement<Microseconds> scope, as constructed and destroyed by
   ConcurrentHashJoin: two clock_gettime(CLOCK_MONOTONIC) calls plus one ProfileEvents::increment
   (a sched_getcpu and three relaxed fetch_adds down the counter chain). */
#define _GNU_SOURCE
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static _Atomic uint64_t sink[3][64];

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t one_timer(void)
{
    uint64_t start = now_ns();          /* Stopwatch ctor */
    uint64_t stop = now_ns();           /* Stopwatch::stop in the dtor */
    uint64_t us = (stop - start) / 1000;
    int cpu = sched_getcpu();           /* PerCPU::getCurrentCPU */
    if (cpu < 0) cpu = 0;
    for (int level = 0; level < 3; ++level)
        atomic_fetch_add_explicit(&sink[level][cpu & 63], us + 1, memory_order_relaxed);
    return stop;
}

int main(int argc, char ** argv)
{
    long iters = argc > 1 ? atol(argv[1]) : 2000000;
    /* Warm the vDSO page and the cache lines. */
    for (long i = 0; i < 100000; ++i) one_timer();
    uint64_t t0 = now_ns();
    for (long i = 0; i < iters; ++i) one_timer();
    uint64_t t1 = now_ns();
    /* Two bare clock reads, for reference: the timer minus this is the increment. */
    uint64_t c0 = now_ns();
    for (long i = 0; i < iters; ++i) { now_ns(); now_ns(); }
    uint64_t c1 = now_ns();
    printf("ns_per_timer\t%.2f\n", (double)(t1 - t0) / (double)iters);
    printf("ns_per_clock_pair\t%.2f\n", (double)(c1 - c0) / (double)iters);
    printf("iters\t%ld\n", iters);
    /* Keep the accumulators alive. */
    uint64_t keep = 0;
    for (int l = 0; l < 3; ++l) for (int c = 0; c < 64; ++c) keep += atomic_load(&sink[l][c]);
    fprintf(stderr, "checksum %llu\n", (unsigned long long)keep);
    return 0;
}
EOF
    if ! cc -O2 -o "${bin}" "${src}" 2>"${OUT}/timer_cost.build.log"; then
        echo "0" > "${out}"
        echo "# ProfileEventTimeIncrement calibration failed to compile; see ${OUT}/timer_cost.build.log" >&2
        echo 0
        return 0
    fi
    "${WRAP}" -- "${bin}" "${TIMER_ITERS:-2000000}" > "${out}" 2>/dev/null || \
        "${bin}" "${TIMER_ITERS:-2000000}" > "${out}" 2>/dev/null || true
    ns="$(awk -F'\t' '$1 == "ns_per_timer" { print $2 }' "${out}" 2>/dev/null || true)"
    echo "${ns:-0}"
}

## ---------------------------------------------------------------------------------------------
## Query shapes
##
## All from `numbers()` / `numbers_mt()`. `numbers_mt` produces max_threads streams, which is what
## makes the build side parallel; plain `numbers` is one stream. Blocks are exactly max_block_size
## rows, so the block count of every shape below is known and not estimated.
##
##   sql_build_only <rows> <keyexpr>
##       A one-row probe side, so the query is a build. Per-build-block effects (D10, D14's build
##       timers, D20) are undiluted, and `FillingRightJoinSide.elapsed_us` is the whole join.
##
##   sql_probe_heavy <build_rows> <probe_rows> <keyexpr>
##       A small build side and a large probe side, both narrow. For D14's probe timers and D19's
##       per-probe-block lock: the number of probe blocks is probe_rows / max_block_size.
##
##   sql_str_build <rows> <width>
##       A String key, i.e. one `keyHolderPersistKey` copy into an arena per distinct key. The only
##       shape whose memory footprint is dominated by the arenas (D23).
## ---------------------------------------------------------------------------------------------
sql_build_only() {   # sql_build_only <rows> <key-expression over `number`>
    local rows="$1" key="${2:-number}"
    # `max(r.k)` rather than `count()`: it references a right-side column, so no plan rewrite can
    # decide the join is redundant and drop the build side. With one probe row it costs nothing.
    echo "SELECT max(r.k) FROM (SELECT 0 AS k) AS p
          LEFT JOIN (SELECT ${key} AS k FROM numbers_mt(${rows})) AS r ON p.k = r.k"
}

sql_probe_heavy() {   # sql_probe_heavy <build_rows> <probe_rows> [build-key-expr] [probe-key-expr]
    local brows="$1" prows="$2" bkey="${3:-number}" pkey="${4:-number % ${1}}"
    echo "SELECT count() FROM (SELECT ${pkey} AS k FROM numbers_mt(${prows})) AS p
          INNER JOIN (SELECT ${bkey} AS k FROM numbers(${brows})) AS r ON p.k = r.k"
}

sql_str_build() {   # sql_str_build <rows> <width>
    local rows="$1" width="${2:-48}"
    echo "SELECT max(r.k) FROM (SELECT '' AS k) AS p
          LEFT JOIN (SELECT rightPad(hex(sipHash128(number)), ${width}, 'x') AS k FROM numbers_mt(${rows})) AS r ON p.k = r.k"
}

## ---------------------------------------------------------------------------------------------
## Shape validation
##
## Every shape here is a subquery join whose build side the planner could in principle rewrite away
## - and a cell whose build side silently disappeared would report a beautifully stable number that
## measures nothing. So each pass validates its shapes once: the join object must be the expected
## one, and `JoinBuildTableRowCount` must equal the row count the SQL asked for.
## ---------------------------------------------------------------------------------------------
# sp_check_shape <tag> <sql> <expected_build_rows> [expected_probe_rows]
#   Runs the query once, then reports "ok" or what was wrong. Non-fatal by design: a cell that
#   cannot be validated is worth reporting rather than aborting the whole script.
sp_check_shape() {
    local tag="$1" sql="$2" want_build="$3" want_probe="${4:-}"
    local algo got_build got_probe verdict
    algo="$(sp_algorithm "${sql}")"
    sp_time "${tag}" "${sql}" >/dev/null || true
    sp_flush
    got_build="$(client --query "SELECT ProfileEvents['JoinBuildTableRowCount'] FROM system.query_log
                                 WHERE query_id = '${tag}' AND type = 'QueryFinish' LIMIT 1" 2>/dev/null || echo 0)"
    got_probe="$(client --query "SELECT ProfileEvents['JoinProbeTableRowCount'] FROM system.query_log
                                 WHERE query_id = '${tag}' AND type = 'QueryFinish' LIMIT 1" 2>/dev/null || echo 0)"
    verdict=ok
    [ "${got_build:-0}" = "${want_build}" ] || verdict="BUILD ROWS ${got_build:-0} != ${want_build}"
    if [ -n "${want_probe}" ] && [ "${got_probe:-0}" != "${want_probe}" ]; then
        verdict="${verdict}; PROBE ROWS ${got_probe:-0} != ${want_probe}"
    fi
    printf '  shape %-28s algorithm=%-34s build_rows=%-12s probe_rows=%-12s %s\n' \
        "${tag}" "${algo}" "${got_build:-0}" "${got_probe:-0}" "${verdict}"
}

## ---------------------------------------------------------------------------------------------
## Per-block slope
##
## The whole family is per-BLOCK while nearly every other divergence between the arms - the
## `scatterBlockBySlot` pass of divergence A above all - is per-ROW. Holding the row count fixed
## and shrinking max_block_size multiplies the number of per-block events and leaves the per-row
## work alone, so the SLOPE of time against block count is the per-block cost and the intercept
## absorbs everything per-row. Two points suffice; four make it possible to see whether the
## relationship is actually linear, which is the check that the design is sound.
## ---------------------------------------------------------------------------------------------
# sp_slope <file> <key-columns-as-awk-expr> <blocks-col> <value-col>
#   Least-squares slope of value against block count within each key group, printed as
#   "<key>\t<slope>\t<points>". Values are seconds or microseconds; the slope carries the same unit
#   per block.
sp_slope() {
    local f="$1" keyexpr="$2" bcol="$3" vcol="$4"
    awk -F'\t' -v b="${bcol}" -v v="${vcol}" '
        NR == 1 { next }
        {
            k = '"${keyexpr}"'
            if ($v == "null" || $v == "" || $b + 0 <= 0) next
            n[k]++; sx[k] += $b; sy[k] += $v; sxx[k] += $b * $b; sxy[k] += $b * $v
        }
        END {
            for (k in n) {
                if (n[k] < 2) continue
                den = n[k] * sxx[k] - sx[k] * sx[k]
                if (den == 0) continue
                printf "%s\t%.6g\t%d\n", k, (n[k] * sxy[k] - sx[k] * sy[k]) / den, n[k]
            }
        }' "${f}" | sort
}
