#!/usr/bin/env bash
# Shared harness for the hash-table-type divergence measurements: m_D4, m_D8, m_D9, m_D13, m_D16.
# Sourced, never run directly.
#
# This file is deliberately self-contained rather than sourcing _common.sh, which belongs to the
# build-path measurements (m_A, m_B, m_D1, m_D2, m_D3). Two reasons: that file bakes
# `enable_join_fixed_hash_table_conversion=0` into every query, which would silently disable the
# range* maps that half of D4 is about; and the two sets of scripts are edited independently.
# What is shared with it, deliberately, is the lock file and the server directory layout, so the
# two families cannot run at the same time and do not fight over configs.
#
# Conventions inherited from ../{job_perf,thread_sweep,deep_metrics_norm}.sh:
#   * the 16-vCPU / 32 GiB cgroup comes from ../cgroup_wrap.sh --print-cg;
#   * the server is started from a helper that puts ITSELF into the cgroup before exec, otherwise
#     it sees the host's CPUs and picks a wrong default max_threads;
#   * TCP port 19010 and one shared data directory, so only one script may run at a time;
#   * timings come from `client --time --format=Null`;
#   * `--collect_hash_table_stats_during_joins=0` on both arms for equal-plan comparisons.
#
# Arms. `hash` and `parallel_hash` are measured with clickhouse-baseline (the merge-base build).
# The branch added seven ProfileEventTimeIncrement timers to ConcurrentHashJoin's build and probe
# hot paths (divergence D14); measuring parallel_hash from the branch build would charge the
# baseline arm a clock_gettime pair per block. `unified_hash` only exists in clickhouse-uhj.

set -euo pipefail

MAPS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_HOME="$(dirname "${MAPS_HOME}")"
WORK="${WORK:-/mnt/data/uhj_versions_bench}"
WRAP="${BENCH_HOME}/cgroup_wrap.sh"
VB="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}"
PORT="${PORT:-19010}"
HTTP_PORT="${HTTP_PORT:-18110}"

BIN_BASELINE="${WORK}/bin/clickhouse-baseline"
BIN_UHJ="${WORK}/bin/clickhouse-uhj"
OUTROOT="${WORK}/measure"
SRVROOT="${OUTROOT}/server"
LOCKFILE="${OUTROOT}/.lock"
SYNTH_DB="${SYNTH_DB:-bench_synth}"

REPS="${REPS:-5}"
QUERY_TIMEOUT="${QUERY_TIMEOUT:-1200}"

mkdir -p "${OUTROOT}" "${SRVROOT}"

CUR_ARM=""
CUR_BIN=""
OUT=""
MEASURE_ID="${MEASURE_ID:-maps}"

## --------------------------------------------------------------------------------------------
## Mutual exclusion and output directory
## --------------------------------------------------------------------------------------------

# stop_server kills every process in the cgroup, so two concurrent measurements would destroy
# each other's server. Same lock path as _common.sh so the two families exclude each other too.
maps_take_lock() {
    exec 9>"${LOCKFILE}"
    if ! flock -n 9; then
        echo "another measurement already holds ${LOCKFILE} - refusing to start" >&2
        exit 1
    fi
    echo "$$" >&9
}

maps_init() {   # maps_init <id>
    MEASURE_ID="$1"
    OUT="${OUTROOT}/${MEASURE_ID}"
    mkdir -p "${OUT}"
    RESULT_TSV="${OUT}/timings.tsv"
    STAGE_TSV="${OUT}/stages.tsv"
    QLOG_TSV="${OUT}/qlog.tsv"
    CHECK_TSV="${OUT}/results_check.tsv"
    MAPTYPE_TXT="${OUT}/maptypes.txt"
    printf 'case\talgo\tmax_threads\textra\trep\tseconds\tquery_id\n' > "${RESULT_TSV}"
    printf 'case\talgo\tmax_threads\tresult\n' > "${CHECK_TSV}"
    : > "${MAPTYPE_TXT}"
    RUN_EPOCH="$(date -u +%s)"
}

## --------------------------------------------------------------------------------------------
## Server lifecycle
## --------------------------------------------------------------------------------------------

arm_bin() {
    case "$1" in
        baseline) echo "${BIN_BASELINE}" ;;
        uhj)      echo "${BIN_UHJ}" ;;
        *) echo "unknown arm '$1' (expected baseline|uhj)" >&2; return 1 ;;
    esac
}

algo_arm() {
    case "$1" in
        unified_hash)       echo uhj ;;
        hash|parallel_hash) echo baseline ;;
        *) echo "unknown join_algorithm '$1'" >&2; return 1 ;;
    esac
}

client() { env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" "$@"; }
server_alive() { [ -n "${CUR_BIN}" ] || return 1; client --query 'SELECT 1' </dev/null >/dev/null 2>&1; }
server_pid() { pgrep -f "${CUR_BIN} server" | head -1; }

# The bench config written by ../run_arm.sh declares only query_log. Every script here reads
# per-stage timings out of system.processors_profile_log, so declare that too.
write_server_conf() {
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
    # join_algorithm is passed per query, not baked in here, so one baseline server can serve both
    # the `hash` and the `parallel_hash` arm without a restart.
    #
    # The two LowCardinality write settings are in the profile rather than only on the INSERT
    # because background merges run with the default profile, not with the settings the INSERT
    # used. Left at their defaults (8192 entries, several dictionaries per part allowed), the
    # OPTIMIZE FINAL in synth_table would rewrite a carefully sized one-million-entry dictionary
    # as a hundred and twenty small ones, and the per-block dictionary size that D8, D13 and D16
    # all scale with would become an artifact of the merge rather than a property of the fixture.
    cat > "${dir}/users.xml" <<EOF
<clickhouse>
    <profiles><default>
        <max_memory_usage>0</max_memory_usage>
        <low_cardinality_use_single_dictionary_for_part>1</low_cardinality_use_single_dictionary_for_part>
        <low_cardinality_max_dictionary_size>4000000</low_cardinality_max_dictionary_size>
    </default></profiles>
    <users><default><password></password><networks><ip>::/0</ip></networks>
        <profile>default</profile><quota>default</quota><access_management>1</access_management></default></users>
    <quotas><default><interval><duration>3600</duration></interval></default></quotas>
</clickhouse>
EOF
}

stop_server() {
    local p
    for p in $(cat /sys/fs/cgroup/uhj_versions_bench/run/cgroup.procs 2>/dev/null); do
        case "$(tr '\0' ' ' < "/proc/${p}/cmdline" 2>/dev/null)" in
            *uhj_versions_bench*) kill "${p}" 2>/dev/null || true ;;
        esac
    done
    for _ in $(seq 1 120); do server_alive || break; sleep 1; done
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    fuser -k "${HTTP_PORT}/tcp" 2>/dev/null || true
    sleep 1
}

start_server() {   # start_server <baseline|uhj>
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
    for _ in $(seq 1 240); do server_alive && break; sleep 1; done
    server_alive || { echo "server (${arm}) did not start; see ${dir}/log/" >&2; exit 1; }
    # A server that escaped the cgroup would see all host CPUs and change every max_threads
    # default, so check rather than assume. Expect 16.
    echo "# server up: arm=${arm} bin=$(basename "${CUR_BIN}") default_max_threads=$(client --query "SELECT getSetting('max_threads')" 2>/dev/null || echo '?')"
}

## --------------------------------------------------------------------------------------------
## Settings
##
## MAPS_SETTINGS holds only what never varies across the cases in this family; everything a case
## actually manipulates (join_algorithm, max_threads, the fixed-map conversion, runtime filters,
## block size) is passed at the call site. Nothing appears twice on one command line: the
## ClickHouse client rejects a repeated setting rather than letting the later one win.
##
##   collect_hash_table_stats_during_joins=0  equal reserve policy on both arms (harness rule, D11)
##   query_plan_join_swap_table=false         the right table is always the build side, so a RIGHT
##                                            JOIN really iterates the map the SQL says it builds
##   query_plan_optimize_join_order_limit=0   no join reordering
##   query_plan_convert_join_to_in=0          the join must stay a join
##   parallel_hash_join_threshold=0           parallel_hash is never downgraded to serial hash;
##                                            the branch bypasses that gate for unified_hash (D1),
##                                            and leaving it in would mix D1 into every number here
##   enable_join_runtime_filters=0            on by default, and for exactly the key8/key16/range*
##                                            maps D4 and D9 are about it publishes a shared
##                                            fixed-hash-table filter that can prune the probe
##                                            source. That is a different mechanism with its own
##                                            divergences; the cases that want it pass
##                                            --enable_join_runtime_filters=1 explicitly.
## --------------------------------------------------------------------------------------------
MAPS_SETTINGS=(
    --collect_hash_table_stats_during_joins=0
    --query_plan_join_swap_table=false
    --query_plan_optimize_join_order_limit=0
    --query_plan_convert_join_to_in=0
    --parallel_hash_join_threshold=0
    --enable_join_runtime_filters=0
    --max_memory_usage=0
    --log_processors_profiles=1
)

# Per-query hardware counters, attributed to the query's own threads and landing in
# system.query_log.ProfileEvents. More precise than a server-wide `perf stat` window and it needs
# no iteration counting. Six events, so nothing multiplexes. Populated into MAPS_PERFEV only if
# perfev_available says the kernel lets the server open them.
PERFEV_SETTINGS=(
    --metrics_perf_events_enabled=1
    --metrics_perf_events_list=PerfInstructions,PerfCPUCycles,PerfCacheMisses,PerfBranchMisses,PerfStalledCyclesBackend,PerfDataTLBMisses
)
MAPS_PERFEV=()

flush_logs() { client --query "SYSTEM FLUSH LOGS" >/dev/null 2>&1 || true; }

hr() { printf '%s\n' "-------------------------------------------------------------------------------"; }

# Set WANT_ARM to restrict a re-run to one algorithm, e.g. after fixing one arm's fixture.
want_this_arm() { [ -z "${WANT_ARM:-}" ] || [ "${WANT_ARM}" = "$1" ]; }

perfev_available() {
    local qid="perfev_probe_$$" v
    client --query_id "${qid}" "${PERFEV_SETTINGS[@]}" --format=Null \
        --query "SELECT sum(number) FROM numbers(20000000)" >/dev/null 2>&1 || return 1
    flush_logs
    v="$(client --query "SELECT ProfileEvents['PerfInstructions'] FROM system.query_log
                         WHERE query_id = '${qid}' AND type = 'QueryFinish' LIMIT 1" 2>/dev/null || echo 0)"
    [ -n "${v}" ] && [ "${v}" != "0" ]
}

# Call once after the first server is up. Enables per-query counters when the kernel allows them.
maps_enable_perfev() {
    if perfev_available; then
        MAPS_PERFEV=("${PERFEV_SETTINGS[@]}")
        echo "# per-query hardware counters: enabled"
    else
        MAPS_PERFEV=()
        echo "# per-query hardware counters: unavailable (perf_event_paranoid?); wall time and stage times only"
    fi
}

## --------------------------------------------------------------------------------------------
## Measurement primitives
## --------------------------------------------------------------------------------------------

# _merge_settings <extra...> -> MERGED_SETTINGS
# The ClickHouse client rejects a repeated option outright rather than letting the last one win, so
# a case that wants to override one of the defaults above cannot simply append it. Drop any default
# whose name the caller also supplies, then append the caller's.
MERGED_SETTINGS=()
_merge_settings() {
    local -a base=("${MAPS_SETTINGS[@]}" ${MAPS_PERFEV[@]+"${MAPS_PERFEV[@]}"})
    local -a out=()
    local b e skip
    for b in "${base[@]}"; do
        skip=0
        for e in "$@"; do
            [ "${e%%=*}" = "${b%%=*}" ] && { skip=1; break; }
        done
        [ "${skip}" = 0 ] && out+=("${b}")
    done
    MERGED_SETTINGS=(${out[@]+"${out[@]}"} "$@")
}

# run_point <case> <algo> <max_threads> <db> <sql> [extra client settings...]
# One untimed warm-up, then REPS timed repetitions. Each repetition gets a query_id encoding the
# coordinates so collect_stages and collect_qlog can attribute rows without a side table.
run_point() {
    local case_name="$1" algo="$2" mt="$3" db="$4" sql="$5"; shift 5
    local extra_tag="-"
    [ "$#" -gt 0 ] && { extra_tag="$(printf '%s ' "$@")"; extra_tag="${extra_tag% }"; }
    _merge_settings "$@"

    local args=(--database "${db}" --format=Null "${MERGED_SETTINGS[@]}"
                --join_algorithm="${algo}" --max_threads="${mt}")

    CH_TIMEOUT="${QUERY_TIMEOUT}" timeout "${QUERY_TIMEOUT}" \
        env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        "${args[@]}" --query "${sql}" >/dev/null 2>&1 || true

    local rep t qid line=""
    for rep in $(seq 1 "${REPS}"); do
        qid="${MEASURE_ID}|${case_name}|${algo}|${mt}|${rep}"
        t="$(timeout "${QUERY_TIMEOUT}" env HOME=/tmp TZ=UTC "${CUR_BIN}" client \
                --host 127.0.0.1 --port "${PORT}" "${args[@]}" \
                --query_id "${qid}" --time --query "${sql}" 2>&1 | tail -1)"
        [[ "${t}" =~ ^[0-9]+\.[0-9]+$ ]] || t="null"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${case_name}" "${algo}" "${mt}" "${extra_tag}" "${rep}" "${t}" "${qid}" >> "${RESULT_TSV}"
        line="${line} ${t}"
    done
    printf '  %-30s %-13s mt=%-3s %s\n' "${case_name}" "${algo}" "${mt}" "${line}"
}

# record_result <case> <algo> <max_threads> <db> <sql> [extra settings...]
# Runs the query once for its value rather than its time. Every case in this family is written so
# both arms must return the same scalars; a mismatch means the arms did not run the same query and
# the timings beside it mean nothing.
record_result() {
    local case_name="$1" algo="$2" mt="$3" db="$4" sql="$5"; shift 5
    local val
    _merge_settings "$@"
    val="$(timeout "${QUERY_TIMEOUT}" env HOME=/tmp TZ=UTC "${CUR_BIN}" client \
            --host 127.0.0.1 --port "${PORT}" --database "${db}" --format=TSV \
            "${MERGED_SETTINGS[@]}" --join_algorithm="${algo}" --max_threads="${mt}" \
            --query "${sql}" 2>&1 | tr '\n\t' ';,' | cut -c1-160)"
    printf '%s\t%s\t%s\t%s\n' "${case_name}" "${algo}" "${mt}" "${val}" >> "${CHECK_TSV}"
}

# capture_maptype <case> <algo> <max_threads> <db> <sql> [extra settings...]
# Which hash-table variant was actually built is the single most important thing to confirm before
# reading any number in this family. The cheap, reliable way is HashJoin's own LOG_TEST line, which
# both arms emit at construction:
#     "<instance> Keys: ..., datatype: <Type>, kind: ..., strictness: ..."
# It reaches the client only at send_logs_level=test. parallel_hash prints one line per shard, each
# tagged "(concurrentN)", so the count of distinct instance ids also shows how many maps were built
# -- which is exactly the D9 question. Two more lines are captured when present:
#     "Using a dictionary-aware hash map for the single LowCardinality join key"  (D8/D13/D16)
#     "Converted join hash map to fixed hash map (range: R, keys: K)"             (D4 range* maps)
# The full log is kept so the runner can look for anything the greps missed.
#
# Independent confirmation, for when the log line is not enough: run the query under
#     sudo perf record -F 299 -g -p <server pid>
# and look at the demangled symbol names. The chosen map appears in the mangled
# `insertFromBlockImplTypeCase` and `joinRightColumns` instantiations -- `PartitionedFixedHashMap`
# / `FixedHashMap` / `TwoLevelHashMapTable` and the key type are all in the template arguments.
# ../job_perf.sh already does exactly this recording; point it at one of the case queries.
capture_maptype() {
    local case_name="$1" algo="$2" mt="$3" db="$4" sql="$5"; shift 5
    local raw="${OUT}/logs_${case_name}_${algo}_mt${mt}.txt"
    _merge_settings "$@"
    # send_logs_level=test on a hundred-million-row query emits hundreds of megabytes of
    # part-reading chatter. Keep only the lines that say something about the map.
    timeout "${QUERY_TIMEOUT}" env HOME=/tmp TZ=UTC "${CUR_BIN}" client \
        --host 127.0.0.1 --port "${PORT}" --database "${db}" --format=Null \
        "${MERGED_SETTINGS[@]}" --join_algorithm="${algo}" --max_threads="${mt}" \
        --send_logs_level=test --query "${sql}" 2>&1 >/dev/null \
        | grep -aE 'datatype:|dictionary-aware hash map|Converted join hash map|concurrent[0-9]' \
        > "${raw}" || true
    {
        printf '%-30s %-13s mt=%-3s : ' "${case_name}" "${algo}" "${mt}"
        local types insts
        types="$(sed -n 's/.*datatype: \([a-z0-9_]*\).*/\1/p' "${raw}" | sort -u | paste -sd, - )"
        insts="$(grep -c 'datatype:' "${raw}" 2>/dev/null || echo 0)"
        printf '%s (%s map construction log lines)' "${types:-UNKNOWN - see $(basename "${raw}")}" "${insts}"
        grep -q 'dictionary-aware hash map' "${raw}" && printf ' [dictionary-aware]'
        sed -n 's/.*\(Converted join hash map to fixed hash map ([^)]*)\).*/ [\1]/p' "${raw}" | sort -u | tr -d '\n'
        printf '\n'
    } >> "${MAPTYPE_TXT}"
}

## --------------------------------------------------------------------------------------------
## Log extraction. Call from the LAST arm to run (uhj): both system log tables live in the shared
## data directory, and the branch build can read parts written by the merge-base build while the
## reverse is not guaranteed.
##
## Each script none the less starts baseline first, on a data directory whose system logs the
## branch may have written last. That is the unsafe direction, and it is deliberate: a system log
## table whose schema no longer matches is renamed aside and recreated by the server at startup
## rather than failing, and every query here is filtered by RUN_EPOCH, so at worst the previous
## script's rows become unreadable. If `stages.tsv` ever comes back empty, check the baseline
## server's startup log for a `query_log` / `processors_profile_log` rename before suspecting the
## measurement.
## --------------------------------------------------------------------------------------------

collect_stages() {
    flush_logs
    client --query "
        SELECT
            splitByChar('|', query_id)[2]               AS case_name,
            splitByChar('|', query_id)[3]               AS algo,
            toUInt32(splitByChar('|', query_id)[4])     AS max_threads,
            toUInt32(splitByChar('|', query_id)[5])     AS rep,
            sumIf(elapsed_us, name = 'FillingRightJoinSide')                                AS build_us,
            countIf(name = 'FillingRightJoinSide')                                          AS build_streams,
            sumIf(elapsed_us, name = 'JoiningTransform')                                    AS probe_us,
            sumIf(elapsed_us, name LIKE '%NonJoined%' OR name LIKE 'DelayedJoinedBlocks%')   AS nonjoined_us,
            countIf(name LIKE '%NonJoined%' OR name LIKE 'DelayedJoinedBlocks%')             AS nonjoined_streams,
            sum(elapsed_us)                                                                  AS all_us
        FROM system.processors_profile_log
        WHERE query_id LIKE '${MEASURE_ID}|%' AND event_time >= toDateTime(${RUN_EPOCH})
        GROUP BY query_id
        ORDER BY case_name, algo, max_threads, rep
        FORMAT TSVWithNames" > "${STAGE_TSV}" 2>/dev/null || echo "collect_stages: failed" >&2
}

collect_qlog() {
    flush_logs
    client --query "
        SELECT
            splitByChar('|', query_id)[2]           AS case_name,
            splitByChar('|', query_id)[3]           AS algo,
            toUInt32(splitByChar('|', query_id)[4]) AS max_threads,
            count()                                 AS runs,
            round(min(query_duration_ms) / 1000, 4) AS best_sec,
            any(ProfileEvents['JoinBuildTableRowCount'])                AS build_rows,
            any(ProfileEvents['JoinProbeTableRowCount'])                AS probe_rows,
            any(ProfileEvents['JoinResultRowCount'])                    AS result_rows,
            toUInt64(median(ProfileEvents['PerfInstructions']))         AS instructions,
            toUInt64(median(ProfileEvents['PerfCPUCycles']))            AS cycles,
            toUInt64(median(ProfileEvents['PerfCacheMisses']))          AS cache_misses,
            toUInt64(median(ProfileEvents['PerfDataTLBMisses']))        AS dtlb_misses,
            toUInt64(median(ProfileEvents['PerfStalledCyclesBackend'])) AS stall_backend,
            if(build_rows > 0, round(instructions / build_rows, 2), 0)  AS instr_per_build_row,
            if(probe_rows > 0, round(instructions / probe_rows, 2), 0)  AS instr_per_probe_row,
            toUInt64(median(memory_usage))                              AS memory_usage
        FROM system.query_log
        WHERE type = 'QueryFinish' AND query_id LIKE '${MEASURE_ID}|%'
          AND event_time >= toDateTime(${RUN_EPOCH})
        GROUP BY case_name, algo, max_threads
        ORDER BY case_name, algo, max_threads
        FORMAT TSVWithNames" > "${QLOG_TSV}" 2>/dev/null || echo "collect_qlog: failed" >&2
}

## --------------------------------------------------------------------------------------------
## Reporting
## --------------------------------------------------------------------------------------------

# median of a whitespace-separated list, in awk
_MEDIAN_AWK='function med(s,   n,a,i,j,t) { n=split(s,a," "); if(!n) return 0;
    for(i=1;i<n;i++) for(j=i+1;j<=n;j++) if(a[j]+0<a[i]+0){t=a[i];a[i]=a[j];a[j]=t}
    return a[int((n+1)/2)]+0 }'

summarize() {
    echo
    echo "=== ${MEASURE_ID}: wall clock, seconds (min / median of ${REPS}) ==="
    awk -F'\t' "${_MEDIAN_AWK}"'
        NR>1 && $6!="null" { k=$1"\t"$2"\t"$3; v[k]=v[k]" "$6; if (mn[k]=="" || $6+0<mn[k]+0) mn[k]=$6 }
        END { for (k in v) { split(k,a,"\t"); printf "%-30s %-13s %4s %10.4f %10.4f\n", a[1], a[2], a[3], mn[k], med(v[k]) } }' \
        "${RESULT_TSV}" | sort | awk 'BEGIN{printf "%-30s %-13s %4s %10s %10s\n","case","algo","mt","min","median"} {print}'

    if [ -s "${STAGE_TSV}" ] && [ "$(wc -l < "${STAGE_TSV}")" -gt 1 ]; then
        echo
        echo "=== ${MEASURE_ID}: stage time, microseconds (median over reps, summed over streams) ==="
        awk -F'\t' "${_MEDIAN_AWK}"'
            NR>1 { k=$1"\t"$2"\t"$3; b[k]=b[k]" "$5; bs[k]=$6; p[k]=p[k]" "$7; n[k]=n[k]" "$8; ns[k]=$9 }
            END { for (k in b) { split(k,a,"\t");
                printf "%-30s %-13s %4s %12d %7s %12d %12d %7s\n", a[1],a[2],a[3], med(b[k]), bs[k], med(p[k]), med(n[k]), ns[k] } }' \
            "${STAGE_TSV}" | sort | awk 'BEGIN{printf "%-30s %-13s %4s %12s %7s %12s %12s %7s\n","case","algo","mt","build_us","bstrm","probe_us","nonjoin_us","njstrm"} {print}'
    fi

    if [ -s "${QLOG_TSV}" ] && [ "$(wc -l < "${QLOG_TSV}")" -gt 1 ]; then
        echo
        echo "=== ${MEASURE_ID}: query_log counters (median over reps) ==="
        column -t -s $'\t' "${QLOG_TSV}" 2>/dev/null || cat "${QLOG_TSV}"
    fi

    echo
    echo "=== ${MEASURE_ID}: hash-table variant actually built ==="
    cat "${MAPTYPE_TXT}" 2>/dev/null || true

    echo
    echo "=== ${MEASURE_ID}: cross-arm result agreement ==="
    awk -F'\t' 'NR>1 { k=$1"\t"$3; if (!(k in v)) v[k]=$4; else if (v[k]!=$4) bad[k]=1 }
        END { n=0; for (k in bad) { split(k,a,"\t"); printf "MISMATCH  %s  mt=%s  (arms did not run the same query)\n", a[1], a[2]; n++ }
              if (!n) print "all arms returned identical results" }' "${CHECK_TSV}"

    echo
    echo "results: ${OUT}"
}

## --------------------------------------------------------------------------------------------
## perf (optional, PERF=1). The per-query counters above are the primary instrument; this exists
## for the memory-behaviour questions where a wider event set helps. Two 6-event groups so nothing
## multiplexes; cpu_cycles and inst_retired are in both so the passes can be cross-checked.
## --------------------------------------------------------------------------------------------
PERF_CORE='{cpu_cycles,inst_retired,stall_backend,stall_backend_mem,br_mis_pred_retired,mem_access}'
PERF_MEM='{cpu_cycles,inst_retired,l1d_cache_refill,ll_cache_miss_rd,dtlb_walk,mem_access}'

# perf_point <tag> <events> <db> <sql> <algo> <max_threads> [extra settings...]
# Loops the query for PERF_SECONDS while perf stat counts the server, recording the iteration count
# so every counter can be reported per query. Same shape as ../deep_metrics_norm.sh.
perf_point() {
    [ "${PERF:-0}" = "1" ] || return 0
    local tag="$1" events="$2" db="$3" sql="$4" algo="$5" mt="$6"; shift 6
    local secs="${PERF_SECONDS:-25}" base="${OUT}/perf_${tag}" cf pid c0 c1
    cf="${base}.itercount"; echo 0 > "${cf}"
    _merge_settings "$@"
    local -a pargs=("${MERGED_SETTINGS[@]}")
    (
        n=0; end=$((SECONDS + secs + 8))
        while [ ${SECONDS} -lt ${end} ]; do
            env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
                --database "${db}" --format=Null "${pargs[@]}" \
                --join_algorithm="${algo}" --max_threads="${mt}" \
                --query "${sql}" >/dev/null 2>&1 && n=$((n+1))
            echo "${n}" > "${cf}"
        done
    ) &
    local loop=$!
    sleep 3
    c0="$(cat "${cf}" 2>/dev/null || echo 0)"
    pid="$(server_pid)"
    if [ -n "${pid}" ]; then
        sudo perf stat -x, -p "${pid}" -e "${events}" -- sleep "${secs}" 2> "${base}.perf.csv" || true
    else
        echo "server pid not found" > "${base}.perf.csv"; sleep "${secs}"
    fi
    c1="$(cat "${cf}" 2>/dev/null || echo 0)"
    kill "${loop}" 2>/dev/null || true; wait "${loop}" 2>/dev/null || true
    echo "$((c1 - c0))" > "${base}.iters"
    echo "  perf ${tag}: $((c1 - c0)) iterations in ${secs}s -> ${base}.perf.csv"
}

## --------------------------------------------------------------------------------------------
## Synthetic fixtures
##
## Idempotent: a table is left alone when its row count already matches, and rebuilt otherwise.
## Every m_D*.sh calls ensure_synth_maps itself, so the measurements work even if setup_synth.sh
## was never run.
##
## All tables are ORDER BY tuple(): rows keep the order numbers() produced them in, so keys stay
## interleaved across blocks. Sorting by the join key would put few distinct keys in each block and
## would flatter both the LowCardinality dictionary cache and fixed-map locality.
##
## Total footprint with the default row counts is well under 20 GB on disk.
## --------------------------------------------------------------------------------------------

D9_PROBE_ROWS="${D9_PROBE_ROWS:-100000000}"
D9_BUILD_ROWS="${D9_BUILD_ROWS:-50000000}"
LC_PROBE_ROWS="${LC_PROBE_ROWS:-50000000}"
LC_BUILD_ROWS="${LC_BUILD_ROWS:-50000000}"
LC_SWEEP_ROWS="${LC_SWEEP_ROWS:-20000000}"
KEYS_BUILD_ROWS="${KEYS_BUILD_ROWS:-50000000}"

synth_rows() { client --query "SELECT count() FROM ${SYNTH_DB}.$1" 2>/dev/null || echo -1; }

# synth_table <name> <columns> <expected rows> <select> [insert settings...]
synth_table() {
    local name="$1" cols="$2" want="$3" sel="$4"; shift 4
    local have
    client --query "CREATE TABLE IF NOT EXISTS ${SYNTH_DB}.${name} (${cols}) ENGINE = MergeTree ORDER BY tuple()" >/dev/null
    have="$(synth_rows "${name}")"
    if [ "${have}" = "${want}" ]; then
        return 0
    fi
    echo "#   populating ${SYNTH_DB}.${name} (have=${have}, want=${want})"
    client --query "TRUNCATE TABLE ${SYNTH_DB}.${name}" >/dev/null
    timeout 7200 env HOME=/tmp TZ=UTC "${CUR_BIN}" client --host 127.0.0.1 --port "${PORT}" \
        --max_insert_threads=8 --max_memory_usage=0 "$@" \
        --query "INSERT INTO ${SYNTH_DB}.${name} ${sel}" >/dev/null
    client --query "OPTIMIZE TABLE ${SYNTH_DB}.${name} FINAL" >/dev/null 2>&1 || true
    have="$(synth_rows "${name}")"
    [ "${have}" = "${want}" ] || { echo "failed to populate ${name}: ${have} != ${want}" >&2; exit 1; }
}

# A LowCardinality string of total width W drawn from a dictionary of D entries:
# 'k' + a zero-padded 9-digit index + (W-10) filler characters.
lc_key() { printf "concat('k', leftPad(toString(number %% %s), 9, '0'), repeat('x', %s))" "$1" "$(($2 - 10))"; }

# Insert settings that pin one dictionary per part, sized to hold every distinct value. Left at the
# default the writer starts a fresh dictionary every 8192 entries, and the per-block dictionary
# size -- which is precisely what D13 pays for and what D8's crossover depends on -- would be an
# artifact of that default instead of a property of the data.
lc_ins() { echo "--low_cardinality_use_single_dictionary_for_part=1 --low_cardinality_max_dictionary_size=$(( $1 * 2 + 8192 ))"; }

ensure_synth_maps() {
    echo "# ensuring ${SYNTH_DB} fixtures for the map-family measurements"
    client --query "CREATE DATABASE IF NOT EXISTS ${SYNTH_DB}" >/dev/null

    ## ---- D4: build sides for the fixed / direct-addressed maps ------------------------------
    ##
    ## One full traversal of a FixedHashMap costs `end - begin` cell probes. Baseline keeps the
    ## min/max optimisation, so begin = buf+min and end = buf+max+1. `FixedRangeStorage`'s
    ## constructor calls disableMinMaxOptimization() permanently, so on UHJ begin walks forward
    ## from cell 0 until it finds the first populated cell and end is buf+NUM_CELLS. Two separate
    ## penalties follow, and the tables below separate them:
    ##
    ##   leading  - firstPopulatedCell()'s linear search. Paid by *every* stream that touches the
    ##              map, including the RIGHT/FULL non-joined streams that own no iteration bucket
    ##              and return immediately afterwards, so it multiplies by stream count.
    ##   trailing - the walk from the last real key to NUM_CELLS. Paid once, by the owning stream.
    ##
    ##   table                key     rows   map              lead+trail wasted (uhj)   base
    ##   d4_dim_k16_top64     UInt16     64   key16            65472 + 0                   64
    ##   d4_dim_k16_bot64     UInt16     64   key16                0 + 65472               64
    ##   d4_dim_k16_top4096   UInt16   4096   key16            61440 + 0                 4096
    ##   d4_dim_k16_full      UInt16  65536   key16                0 + 0     (control)  65536
    ##   d4_dim_k8_top16      UInt8      16   key8               240 + 0                   16
    ##   d4_dim_k8_full       UInt8     256   key8                 0 + 0     (control)    256
    ##   d4_dim_u64_r257      UInt64    257   range16_key64        0 + 65279              257
    ##   d4_dim_u64_r65k      UInt64  65536   range16_key64        0 + 0     (control)  65536
    ##   d4_dim_u64_r131k     UInt64  65537   range17_key64        0 + 65535            65537
    ##   d4_dim_u64_r262k     UInt64  65536   range18_key64        0 + 131071          131073
    ##   d4_dim_u64_sparse    UInt64   4096   key64 (no conv.)   n/a         (control)    n/a
    ##
    ## The range* rows need care. tryConvertToFixedHashMapImpl stores `key - min_key`, so a
    ## converted map is always populated from cell 0 and only the trailing penalty exists; the
    ## key8/key16 maps store the raw key, which is why placing keys at the top of the key space
    ## is what makes the leading penalty appear at all. The buffer is the next power of two among
    ## {2^8, 2^16, 2^17, 2^18} -- there is no range9..range15 -- so 257 keys land in a 65536-cell
    ## buffer, the largest *ratio* the conversion can produce (255x). d4_dim_u64_r262k is the
    ## largest *absolute* gap the 25% fill guard permits: 65536 keys two apart span 131073, which
    ## overflows range17 into a 262144-cell range18 buffer, and 262144 > 65536*4 is false by
    ## exactly one cell, so the guard passes. d4_dim_u64_sparse spans 131041 with only 4096 keys,
    ## fails the guard, and stays a plain key64 -- the control that shows the effect belongs to
    ## the fixed map and not to the shape of the data.
    synth_table d4_dim_k16_top64   'k UInt16, v UInt64'     64 "SELECT toUInt16(65472 + number), number FROM numbers(64)"
    synth_table d4_dim_k16_bot64   'k UInt16, v UInt64'     64 "SELECT toUInt16(number), number FROM numbers(64)"
    synth_table d4_dim_k16_top4096 'k UInt16, v UInt64'   4096 "SELECT toUInt16(61440 + number), number FROM numbers(4096)"
    synth_table d4_dim_k16_full    'k UInt16, v UInt64'  65536 "SELECT toUInt16(number), number FROM numbers(65536)"
    synth_table d4_dim_k8_top16    'k UInt8, v UInt64'      16 "SELECT toUInt8(240 + number), number FROM numbers(16)"
    synth_table d4_dim_k8_full     'k UInt8, v UInt64'     256 "SELECT toUInt8(number), number FROM numbers(256)"
    synth_table d4_dim_u64_r257    'k UInt64, v UInt64'    257 "SELECT toUInt64(1000000 + number), number FROM numbers(257)"
    synth_table d4_dim_u64_r65k    'k UInt64, v UInt64'  65536 "SELECT toUInt64(1000000 + number), number FROM numbers(65536)"
    synth_table d4_dim_u64_r131k   'k UInt64, v UInt64'  65537 "SELECT toUInt64(1000000 + number), number FROM numbers(65537)"
    synth_table d4_dim_u64_r262k   'k UInt64, v UInt64'  65536 "SELECT toUInt64(1000000 + number * 2), number FROM numbers(65536)"
    synth_table d4_dim_u64_sparse  'k UInt64, v UInt64'   4096 "SELECT toUInt64(1000000 + number * 32), number FROM numbers(4096)"

    # 250 distinct UInt16 keys at the top of the key space, 40 rows each. This is the only shape
    # that satisfies rightTableCanBeReranged(): <= join_to_sort_maximum_table_rows (10000) and
    # >= join_to_sort_minimum_perkey_rows (40) rows per key. It reaches D4 through a second path,
    # tryRerangeRightTableDataImpl's forEachMapped, on an INNER join instead of a RIGHT one.
    synth_table d4_rerange_k16 'k UInt16, v UInt64' 10000 \
        "SELECT toUInt16(65286 + (number % 250)), number FROM numbers(10000)"

    # Probes. The *_nomatch probes leave every build row unmatched, so a RIGHT join's non-joined
    # stream has to walk the whole map and emit every key: iteration is then the entire query.
    # Their key values are chosen to miss every dimension above except the two `_full` controls,
    # where by construction no UInt8/UInt16 value can miss.
    synth_table d4_probe_k8_nomatch  'k UInt8'  4 "SELECT toUInt8(number) FROM numbers(4)"
    synth_table d4_probe_k16_nomatch 'k UInt16' 4 "SELECT toUInt16(100 + number) FROM numbers(4)"
    synth_table d4_probe_u64_nomatch 'k UInt64' 4 "SELECT toUInt64(1 + number) FROM numbers(4)"
    synth_table d4_probe_k16_10m 'k UInt16' 10000000 "SELECT toUInt16(number % 65536) FROM numbers_mt(10000000)"
    synth_table d4_probe_u64_1m  'k UInt64'  1000000 "SELECT toUInt64(1000000 + (number % 257)) FROM numbers_mt(1000000)"

    ## ---- D9: one-and-two-byte keys ----------------------------------------------------------
    ##
    ## Baseline chooseMethod(..., use_two_level_maps=true) has no two-level form for key8/key16
    ## (HashJoin.cpp:418, `default: return type`), so parallel_hash's shards each keep a private
    ## single-level FixedHashMap, twoLevelMapIsUsed() is false, and every probe block goes through
    ## dispatchBlock. UHJ has one shared PartitionedFixedHashMap and never scatters the probe.
    ## d9_*_k64 carries the same 256 values eight bytes wide: with
    ## enable_join_fixed_hash_table_conversion=0 that pair is key64 / two_level_key64 on both arms,
    ## so D9 and D4 are both off and whatever gap remains belongs to the rest of the fork.
    synth_table d9_dim_k8    'k UInt8, v UInt64'    256   "SELECT toUInt8(number), number FROM numbers(256)"
    synth_table d9_dim_k16   'k UInt16, v UInt64'   65536 "SELECT toUInt16(number), number FROM numbers(65536)"
    synth_table d9_dim_k64   'k UInt64, v UInt64'   256   "SELECT toUInt64(number), number FROM numbers(256)"
    synth_table d9_probe_k8  'k UInt8'  "${D9_PROBE_ROWS}" "SELECT toUInt8(number % 256) FROM numbers_mt(${D9_PROBE_ROWS})"
    synth_table d9_probe_k16 'k UInt16' "${D9_PROBE_ROWS}" "SELECT toUInt16(number % 65536) FROM numbers_mt(${D9_PROBE_ROWS})"
    synth_table d9_probe_k64 'k UInt64' "${D9_PROBE_ROWS}" "SELECT toUInt64(number % 256) FROM numbers_mt(${D9_PROBE_ROWS})"
    synth_table d9_probe_k8_small 'k UInt8' 1000000 "SELECT toUInt8(number % 256) FROM numbers(1000000)"
    # Build-heavy: every block carries all 256 keys, so every build block has rows for every slot
    # -- the shape where the shared PartitionedFixedHashMap must take all N bucket locks for every
    # block while parallel_hash's shards each fill a private FixedHashMap.
    synth_table d9_build_k8  'k UInt8, v UInt64' "${D9_BUILD_ROWS}" "SELECT toUInt8(number % 256), number FROM numbers_mt(${D9_BUILD_ROWS})"

    ## ---- D8 / D16: LowCardinality(String) ---------------------------------------------------
    ##
    ## Only String and FixedString dictionaries reach the dictionary-aware map:
    ## tryGetLowCardinalityMethod rejects numeric nested types. Dictionary size drives the index
    ## width (UInt8 up to 256 entries, UInt16 up to 65536, UInt32 beyond) and the size of the
    ## per-getter visit/mapped/offset caches, and it decides whether probe-side dedup can pay for
    ## itself: once the dictionary is larger than a block, almost every cache slot is touched at
    ## most once and the cache is pure overhead.
    synth_table lc_dim_d16   'k LowCardinality(String), v UInt64'      16 "SELECT $(lc_key 16 48), number FROM numbers(16)"           $(lc_ins 16)
    synth_table lc_dim_d1k   'k LowCardinality(String), v UInt64'    1000 "SELECT $(lc_key 1000 48), number FROM numbers(1000)"       $(lc_ins 1000)
    synth_table lc_dim_d100k 'k LowCardinality(String), v UInt64'  100000 "SELECT $(lc_key 100000 48), number FROM numbers(100000)"   $(lc_ins 100000)
    synth_table lc_dim_d1m   'k LowCardinality(String), v UInt64' 1000000 "SELECT $(lc_key 1000000 48), number FROM numbers(1000000)" $(lc_ins 1000000)
    synth_table str_dim_d1k  'k String, v UInt64'                    1000 "SELECT $(lc_key 1000 48), number FROM numbers(1000)"

    synth_table lc_probe_d16   'k LowCardinality(String)' "${LC_PROBE_ROWS}" "SELECT $(lc_key 16 48) FROM numbers_mt(${LC_PROBE_ROWS})"      $(lc_ins 16)
    synth_table lc_probe_d1k   'k LowCardinality(String)' "${LC_PROBE_ROWS}" "SELECT $(lc_key 1000 48) FROM numbers_mt(${LC_PROBE_ROWS})"    $(lc_ins 1000)
    synth_table lc_probe_d100k 'k LowCardinality(String)' "${LC_PROBE_ROWS}" "SELECT $(lc_key 100000 48) FROM numbers_mt(${LC_PROBE_ROWS})"  $(lc_ins 100000)
    synth_table lc_probe_d1m   'k LowCardinality(String)' "${LC_PROBE_ROWS}" "SELECT $(lc_key 1000000 48) FROM numbers_mt(${LC_PROBE_ROWS})" $(lc_ins 1000000)
    # Plain-String twin of lc_probe_d1k / lc_dim_d1k: the shape parallel_hash ends up
    # materialising anyway, so the dictionary path can be priced against the materialised path
    # without attributing the difference to the arm.
    synth_table str_probe_d1k  'k String' "${LC_PROBE_ROWS}" "SELECT $(lc_key 1000 48) FROM numbers_mt(${LC_PROBE_ROWS})"

    # D16 build sides. Same dictionary (1000 entries) and same row count, two key widths. The
    # divergent branch in UHJ's emplaceKey costs one extra getIndexAt per build row always, and an
    # extra getKeyHolder plus an extra full string hash only when the dictionary has no saved
    # hash. An extra string hash scales with key width; an extra index decode does not. Comparing
    # w16 against w48 therefore tells the two apart without instrumenting anything.
    synth_table lc_build_w16_d1k 'k LowCardinality(String)' "${LC_BUILD_ROWS}" "SELECT $(lc_key 1000 16) FROM numbers_mt(${LC_BUILD_ROWS})" $(lc_ins 1000)
    synth_table lc_build_w48_d1k 'k LowCardinality(String)' "${LC_BUILD_ROWS}" "SELECT $(lc_key 1000 48) FROM numbers_mt(${LC_BUILD_ROWS})" $(lc_ins 1000)
    synth_table str_build_w48_d1k 'k String'                "${LC_BUILD_ROWS}" "SELECT $(lc_key 1000 48) FROM numbers_mt(${LC_BUILD_ROWS})"

    ## ---- D13: per-slot key-getter construction ----------------------------------------------
    ##
    ## shareKeyGetterAcrossBuckets is true only for key getters declaring
    ## reads_whole_block_at_construction, and the only type in the tree that declares it is
    ## HashMethodKeysFixed (HashMethod.h:411) -- the composite fixed-width key getter, whose
    ## constructor packs the whole block into `prepared_keys`. LowCardinalityKeyGetterForJoin does
    ## NOT declare it, so its dictionary-sized caches are rebuilt per slot per block.
    ##
    ## lc_sweep_* is therefore the *unshared* arm of D13: rows are constant and only the
    ## dictionary grows, so any build-time growth is per-slot constructor cost and nothing else.
    ## str_sweep_* is its shared-nothing control: a plain String getter's constructor is O(1), so
    ## its build time must not move with dictionary size at all.
    synth_table lc_sweep_d1k   'k LowCardinality(String)' "${LC_SWEEP_ROWS}" "SELECT $(lc_key 1000 48) FROM numbers_mt(${LC_SWEEP_ROWS})"    $(lc_ins 1000)
    synth_table lc_sweep_d10k  'k LowCardinality(String)' "${LC_SWEEP_ROWS}" "SELECT $(lc_key 10000 48) FROM numbers_mt(${LC_SWEEP_ROWS})"   $(lc_ins 10000)
    synth_table lc_sweep_d100k 'k LowCardinality(String)' "${LC_SWEEP_ROWS}" "SELECT $(lc_key 100000 48) FROM numbers_mt(${LC_SWEEP_ROWS})"  $(lc_ins 100000)
    synth_table lc_sweep_d1m   'k LowCardinality(String)' "${LC_SWEEP_ROWS}" "SELECT $(lc_key 1000000 48) FROM numbers_mt(${LC_SWEEP_ROWS})" $(lc_ins 1000000)
    synth_table str_sweep_d1k  'k String' "${LC_SWEEP_ROWS}" "SELECT $(lc_key 1000 48) FROM numbers_mt(${LC_SWEEP_ROWS})"
    synth_table str_sweep_d1m  'k String' "${LC_SWEEP_ROWS}" "SELECT $(lc_key 1000000 48) FROM numbers_mt(${LC_SWEEP_ROWS})"

    ## The composite-key side, where sharing is actually active. Which of these three is shared in
    ## practice depends on scatterBlockBySlot as much as on the flag: when the key columns total no
    ## more than sizeof(IColumn::Selector::value_type) bytes per row it produces `dense_keys`, and
    ## insertFromBlockImplTypeCase then builds a private getter over the scattered columns and
    ## never consults the shared one. Two UInt32s are exactly 8 bytes and take that path; two
    ## UInt64s are 16 and do not. So:
    ##   d13_build_keys64  (2x UInt32) -> keys64,  packs per slot over dense columns  (bypassed)
    ##   d13_build_keys128 (2x UInt64) -> keys128, packs once per block               (shared)
    ##   d13_build_keys256 (4x UInt64) -> keys256, shared but sizeof(Key)=32 so usePreparedKeys is
    ##                                    false and the constructor is cheap anyway   (null case)
    synth_table d13_build_keys64  'a UInt32, b UInt32' "${KEYS_BUILD_ROWS}" \
        "SELECT toUInt32(number), toUInt32(number * 7 + 1) FROM numbers_mt(${KEYS_BUILD_ROWS})"
    synth_table d13_build_keys128 'a UInt64, b UInt64' "${KEYS_BUILD_ROWS}" \
        "SELECT number, number * 7 + 1 FROM numbers_mt(${KEYS_BUILD_ROWS})"
    synth_table d13_build_keys256 'a UInt64, b UInt64, c UInt64, d UInt64' "${LC_SWEEP_ROWS}" \
        "SELECT number, number * 7 + 1, number * 13 + 2, number * 17 + 3 FROM numbers_mt(${LC_SWEEP_ROWS})"

    ## Four-row probes matching nothing. The build side is materialised in full, the probe costs
    ## nothing, and the join emits no rows, so build cost can be measured without the output side
    ## dominating. 'zzz*' sorts outside every 'k*' dictionary above.
    synth_table lc_nomatch  'k LowCardinality(String)' 4 "SELECT concat('zzz', toString(number)) FROM numbers(4)" $(lc_ins 16)
    synth_table str_nomatch 'k String'                 4 "SELECT concat('zzz', toString(number)) FROM numbers(4)"
    synth_table d13_nomatch_keys64  'a UInt32, b UInt32' 4 "SELECT toUInt32(4000000000 + number), toUInt32(1) FROM numbers(4)"
    synth_table d13_nomatch_keys128 'a UInt64, b UInt64' 4 \
        "SELECT number + toUInt64(1000000000000000000), toUInt64(1) FROM numbers(4)"
    synth_table d13_nomatch_keys256 'a UInt64, b UInt64, c UInt64, d UInt64' 4 \
        "SELECT number + toUInt64(1000000000000000000), toUInt64(1), toUInt64(1), toUInt64(1) FROM numbers(4)"

    echo "# fixtures ready ($(client --query "SELECT formatReadableSize(sum(bytes_on_disk)) FROM system.parts WHERE active AND database='${SYNTH_DB}'" 2>/dev/null) on disk)"
}

## --------------------------------------------------------------------------------------------
## Real-world exposure
##
## Answers, from the live server, the questions every section of SPEC_MAPS.md poses:
##   1. does LowCardinality appear anywhere in job / tpch / tpcds / coffeeshop?  (D8, D13, D16)
##   2. is there any UInt8/UInt16/Enum8/Enum16 column?                           (D9, half of D4)
##   3. which integer columns would tryConvertToFixedHashMapImpl turn into a range* map? (D4)
##   4. does any suite query use a RIGHT or FULL join, the only thing that iterates a map? (D4)
## --------------------------------------------------------------------------------------------
realworld_report() {
    local dst="$1"
    {
        echo "### 1. LowCardinality columns in the four loaded suites"
        client --query "
            SELECT database, table, name, type FROM system.columns
            WHERE database IN ('job','tpch','tpcds','coffeeshop') AND type LIKE '%LowCardinality%'
            ORDER BY database, table, name FORMAT TSVWithNames" 2>&1
        echo "-- an empty result means no LowCardinality anywhere, i.e. zero real-world exposure for D8, D13, D16"
        echo
        echo "### 2. one-and-two-byte integer / Enum columns (would select key8 or key16 directly)"
        client --query "
            SELECT database, table, name, type FROM system.columns
            WHERE database IN ('job','tpch','tpcds','coffeeshop')
              AND (type IN ('UInt8','Int8','UInt16','Int16','Bool') OR type LIKE 'Enum8%' OR type LIKE 'Enum16%')
            ORDER BY database, table, name FORMAT TSVWithNames" 2>&1
        echo "-- an empty result means zero real-world exposure for D9 and for the key8/key16 half of D4"
        echo
        echo "### 3. integer columns that would become a range* FixedHashMap after build"
        echo "-- scanned: 4-and-8-byte integer columns of tables under 4e6 rows (MAX_RANGE is 2^18,"
        echo "-- so a column with more distinct values can never convert)"
        printf 'column\trange\tdistinct\twould_become\n'
        local col db rest tbl cn
        while IFS= read -r col; do
            [ -n "${col}" ] || continue
            db="${col%%.*}"; rest="${col#*.}"; tbl="${rest%%.*}"; cn="${rest#*.}"
            client --query "
                SELECT '${col}', r, u,
                    multiIf(u = 0, 'empty',
                            r > 262144,  'no conversion (range > 2^18)',
                            r <= 256,    'range8_*   (256 cells)',
                            r <= 65536,  'range16_*  (65536 cells)',
                            r <= 131072, if(131072 <= u * 4, 'range17_*  (131072 cells)', 'no conversion (fails 25% fill guard)'),
                            262144 <= u * 4, 'range18_*  (262144 cells)', 'no conversion (fails 25% fill guard)')
                FROM (SELECT toUInt64(max(${cn}) - min(${cn})) + 1 AS r, uniqExact(${cn}) AS u FROM ${db}.${tbl})
                FORMAT TSV" 2>/dev/null || true
        done <<< "$(client --query "
            SELECT concat(c.database, '.', c.table, '.', c.name)
            FROM system.columns AS c
            INNER JOIN (
                SELECT database, table, sum(rows) AS r FROM system.parts
                WHERE active AND database IN ('job','tpch','tpcds','coffeeshop')
                GROUP BY database, table HAVING r < 4000000
            ) AS t ON c.database = t.database AND c.table = t.table
            WHERE c.type IN ('UInt32','Int32','UInt64','Int64')
            ORDER BY 1 FORMAT TSV" 2>/dev/null)"
        echo
        echo "### 4. RIGHT / FULL joins in the four query suites"
        echo "-- D4's iteration penalty is only paid when something walks the whole map, which in"
        echo "-- practice means the RIGHT/FULL non-joined scan. Static grep of the loaded queries:"
        local q n
        for q in job tpch tpcds coffeeshop; do
            n="$(grep -c -iE '\b(right|full)[[:space:]]+(outer[[:space:]]+)?join\b' "${VB}/queries/${q}.sql" 2>/dev/null || echo 0)"
            printf '%s: %s\n' "${q}" "${n}"
            grep -n -iE '\b(right|full)[[:space:]]+(outer[[:space:]]+)?join\b' "${VB}/queries/${q}.sql" 2>/dev/null \
                | sed -E "s|^([0-9]+):.*|  ${q} query \1|" || true
        done
        echo "-- for each hit above, check whether the join key is a single small-range integer;"
        echo "-- a composite key builds a keys64/keys128 map and is not a fixed map at all."
        echo "-- Reading the two tpcds hits statically: q55 joins ON (web.item_sk = store.item_sk)"
        echo "-- AND (web.d_date = store.d_date) and q101 joins ON (customer_sk, item_sk). Both are"
        echo "-- two-column keys, so both build a keys* map and neither can be a fixed map."
        echo
        echo "### 5. is the second D4-exposed iteration path reachable by default?"
        echo "-- tryRerangeRightTableDataImpl walks the whole map with forEachMapped on INNER/LEFT"
        echo "-- joins, but only when allow_experimental_join_right_table_sorting is on:"
        client --query "SELECT name, value, default FROM system.settings
                        WHERE name IN ('allow_experimental_join_right_table_sorting',
                                       'join_to_sort_maximum_table_rows',
                                       'join_to_sort_minimum_perkey_rows')
                        FORMAT TSVWithNames" 2>&1
    } > "${dst}" 2>&1
    echo "# real-world exposure report: ${dst}"
}
