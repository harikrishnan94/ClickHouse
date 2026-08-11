#!/usr/bin/env bash
# Drive ClickBench versions/run-version.sh against a local binary, optionally with
# join_algorithm=unified_hash (the uhj treatment), under the cgroup wrapper.
#
# Keeps the repo runner's timing loop (TRIES, drop_caches, --time --format=Null).
# Only the server start / client path is replaced so we can swap binaries.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${HERE}/work"
VB="${CLICKBENCH_VERSIONS:-/mnt/ch/ClickBench-master/versions}"
DATA="${DATA:-${WORK}/data}"
ARM="${ARM:?set ARM=baseline|uhj}"
ROUND="${ROUND:-1}"
TRIES="${TRIES:-6}"                 # 1 cold + 5 hot — same as run-version.sh
QUERY_TIMEOUT="${QUERY_TIMEOUT:-600}"  # raised vs 100s: constrained 32 GiB / ARM host
LOAD_DATASETS="${LOAD_DATASETS:-coffeeshop tpch tpcds job}"
PHASE="${PHASE:-all}"               # load | bench | all
PORT="${PORT:-19010}"
HTTP_PORT="${HTTP_PORT:-18110}"
# Shared on-disk tables across arms (swap only binary + users.xml). Stronger
# comparability and one load instead of two.
SHARED_DIR="${WORK}/server_shared"
SERVER_DIR="${WORK}/server_${ARM}"
LOG_DIR="${WORK}/logs"
RESULT_DIR="${WORK}/results"
WRAP="${HERE}/cgroup_wrap.sh"
BIN=""
JOIN_ALGO_USERS=""

case "${ARM}" in
    baseline)
        BIN="${WORK}/bin/clickhouse-baseline"
        JOIN_ALGO_USERS=""   # default settings
        ;;
    uhj)
        BIN="${WORK}/bin/clickhouse-uhj"
        # Treatment: unified_hash is the only entry (uhj shorthand in the branch).
        JOIN_ALGO_USERS='<join_algorithm>unified_hash</join_algorithm>'
        ;;
    *) echo "ARM must be baseline or uhj" >&2; exit 1 ;;
esac

[ -x "${BIN}" ] || { echo "missing binary ${BIN}" >&2; exit 1; }
[ -x "${WRAP}" ] || { echo "missing ${WRAP}" >&2; exit 1; }
mkdir -p "${SHARED_DIR}"/{data,tmp,user_files,format_schemas,access} \
         "${SERVER_DIR}"/{log,config.d,users.d} \
         "${LOG_DIR}" "${RESULT_DIR}"

VERSION_LABEL="${ARM}_r${ROUND}"
OUT="${RESULT_DIR}/${VERSION_LABEL}.json"
LOAD_STATS="${LOG_DIR}/${VERSION_LABEL}.loadtimes.tsv"
SERVER_LOG="${LOG_DIR}/${VERSION_LABEL}.server.log"
PIDFILE="${WORK}/clickhouse-server.pid"

# ---- minimal server config (mirrors docker image defaults we care about) ----
cat > "${SERVER_DIR}/config.xml" <<EOF
<clickhouse>
    <logger>
        <level>information</level>
        <log>${SERVER_DIR}/log/server.log</log>
        <errorlog>${SERVER_DIR}/log/server.err.log</errorlog>
        <size>1000M</size>
        <count>3</count>
    </logger>
    <http_port>${HTTP_PORT}</http_port>
    <tcp_port>${PORT}</tcp_port>
    <path>${SHARED_DIR}/data/</path>
    <tmp_path>${SHARED_DIR}/tmp/</tmp_path>
    <user_files_path>${SHARED_DIR}/user_files/</user_files_path>
    <format_schema_path>${SHARED_DIR}/format_schemas/</format_schema_path>
    <access_control_path>${SHARED_DIR}/access/</access_control_path>
    <user_directories>
        <users_xml><path>${SERVER_DIR}/users.xml</path></users_xml>
        <local_directory><path>${SHARED_DIR}/access/</path></local_directory>
    </user_directories>
    <mark_cache_size>5368709120</mark_cache_size>
    <uncompressed_cache_size>0</uncompressed_cache_size>
    <mlock_executable>false</mlock_executable>
    <query_log>
        <database>system</database>
        <table>query_log</table>
        <flush_interval_milliseconds>7500</flush_interval_milliseconds>
    </query_log>
</clickhouse>
EOF

cat > "${SERVER_DIR}/users.xml" <<EOF
<clickhouse>
    <profiles>
        <default>
            <max_memory_usage>0</max_memory_usage>
            ${JOIN_ALGO_USERS}
        </default>
    </profiles>
    <users>
        <default>
            <password></password>
            <networks><ip>::/0</ip></networks>
            <profile>default</profile>
            <quota>default</quota>
            <access_management>1</access_management>
        </default>
    </users>
    <quotas>
        <default>
            <interval>
                <duration>3600</duration>
                <queries>0</queries>
                <errors>0</errors>
                <result_rows>0</result_rows>
                <read_rows>0</read_rows>
                <execution_time>0</execution_time>
            </interval>
        </default>
    </quotas>
</clickhouse>
EOF

client() {
    local to=()
    if [ -n "${CH_TIMEOUT:-}" ]; then to=(timeout "${CH_TIMEOUT}"); fi
    "${to[@]}" env HOME=/tmp TZ=UTC "${BIN}" client --host 127.0.0.1 --port "${PORT}" "$@"
}

drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }
server_alive() { client --query "SELECT 1" </dev/null >/dev/null 2>&1; }

stop_server() {
    if [ -f "${PIDFILE}" ]; then
        local pid; pid="$(cat "${PIDFILE}" 2>/dev/null || true)"
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            for _ in $(seq 1 60); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
            kill -9 "${pid}" 2>/dev/null || true
        fi
        rm -f "${PIDFILE}"
    fi
    # Also kill anything still bound to our ports (stale).
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    fuser -k "${HTTP_PORT}/tcp" 2>/dev/null || true
}

start_server() {
    stop_server
    # Start from a helper that migrates ITSELF into the cgroup before exec, so
    # ClickHouse observes the constrained affinity (max_threads) at startup.
    local cg helper
    cg="$("${WRAP}" --print-cg | awk -F= '/^cg=/{print $2}')"
    helper="${SERVER_DIR}/start_in_cgroup.sh"
    cat > "${helper}" <<EOF
#!/bin/bash
echo \$\$ | sudo tee ${cg}/cgroup.procs >/dev/null
exec "${BIN}" server --config-file="${SERVER_DIR}/config.xml"
EOF
    chmod +x "${helper}"
    nohup "${helper}" >"${SERVER_LOG}" 2>&1 &
    local spid=$!
    echo "${spid}" > "${PIDFILE}"
    local i
    for i in $(seq 1 120); do
        if server_alive; then
            if ! grep -q 'uhj_versions_bench' "/proc/${spid}/cgroup" 2>/dev/null; then
                echo "WARN: server pid ${spid} not in uhj_versions_bench: $(cat /proc/${spid}/cgroup 2>/dev/null)" >&2
            fi
            echo "server up pid=${spid} cgroup=$(cat /proc/${spid}/cgroup) max_threads=$(client --query "SELECT getSetting('max_threads')")" >&2
            return 0
        fi
        sleep 1
    done
    echo "server failed to start; log:" >&2
    tail -40 "${SERVER_LOG}" >&2 || true
    return 1
}

# ---- table map (subset of run-version.sh) ----
declare -A TABLES=(
    [tpch]="nation:tpch_nation.native.zst region:tpch_region.native.zst part:tpch_part.native.zst supplier:tpch_supplier.native.zst partsupp:tpch_partsupp.native.zst customer:tpch_customer.native.zst orders:tpch_orders.native.zst lineitem:tpch_lineitem.native.zst"
    [tpcds]="call_center:tpcds_call_center.native.zst catalog_page:tpcds_catalog_page.native.zst catalog_returns:tpcds_catalog_returns.native.zst catalog_sales:tpcds_catalog_sales.native.zst customer_address:tpcds_customer_address.native.zst customer_demographics:tpcds_customer_demographics.native.zst customer:tpcds_customer.native.zst date_dim:tpcds_date_dim.native.zst household_demographics:tpcds_household_demographics.native.zst income_band:tpcds_income_band.native.zst inventory:tpcds_inventory.native.zst item:tpcds_item.native.zst promotion:tpcds_promotion.native.zst reason:tpcds_reason.native.zst ship_mode:tpcds_ship_mode.native.zst store_returns:tpcds_store_returns.native.zst store_sales:tpcds_store_sales.native.zst store:tpcds_store.native.zst time_dim:tpcds_time_dim.native.zst warehouse:tpcds_warehouse.native.zst web_page:tpcds_web_page.native.zst web_returns:tpcds_web_returns.native.zst web_sales:tpcds_web_sales.native.zst web_site:tpcds_web_site.native.zst"
    [coffeeshop]="fact_sales:coffeeshop_fact_sales.native.zst dim_locations:coffeeshop_dim_locations.native.zst dim_products:coffeeshop_dim_products.native.zst"
    [job]="aka_name:job_aka_name.native.zst aka_title:job_aka_title.native.zst cast_info:job_cast_info.native.zst char_name:job_char_name.native.zst comp_cast_type:job_comp_cast_type.native.zst company_name:job_company_name.native.zst company_type:job_company_type.native.zst complete_cast:job_complete_cast.native.zst info_type:job_info_type.native.zst keyword:job_keyword.native.zst kind_type:job_kind_type.native.zst link_type:job_link_type.native.zst movie_companies:job_movie_companies.native.zst movie_info:job_movie_info.native.zst movie_info_idx:job_movie_info_idx.native.zst movie_keyword:job_movie_keyword.native.zst movie_link:job_movie_link.native.zst name:job_name.native.zst person_info:job_person_info.native.zst role_type:job_role_type.native.zst title:job_title.native.zst"
)
QUERY_ORDER="coffeeshop tpch tpcds job"
VERSION="master"   # create.sh wants a modern version string for DDL

table_exists() { client --database "$1" --query "SELECT 1 FROM $2 LIMIT 0" </dev/null >/dev/null 2>&1; }

run_ddl() {
    local db="$1" stmt rc=0
    while IFS= read -r -d ';' stmt; do
        case "${stmt}" in *[![:space:]]*) : ;; *) continue ;; esac
        if ! client --database "${db}" --query "${stmt}" </dev/null; then
            case "${stmt}" in *[Cc][Rr][Ee][Aa][Tt][Ee]*) rc=1 ;; esac
        fi
    done
    return "${rc}"
}

load_one_dataset() {
    local ds="$1" pair table file ddl t0 cnt
    client --query "CREATE DATABASE IF NOT EXISTS ${ds}" </dev/null
    for pair in ${TABLES[$ds]}; do
        table="${pair%%:*}"; file="${pair##*:}"
        [ -f "${DATA}/${file}" ] || { echo "SKIP ${ds}.${table}: ${file} missing"; continue; }
        if table_exists "${ds}" "${table}"; then
            cnt="$(client --database "${ds}" --query "SELECT count() FROM ${table}" 2>/dev/null | tr -d '\r')"
            [ -n "${cnt}" ] && [ "${cnt}" != "0" ] && { echo "already loaded ${ds}.${table} (${cnt})"; continue; }
        fi
        ddl="$("${VB}/create/create.sh" "${VERSION}" "${ds}" "${table}")"
        echo "=== CREATE ${ds}.${table} ==="
        if ! printf '%s' "${ddl}" | run_ddl "${ds}"; then
            echo "CREATE ${ds}.${table} FAILED"; continue
        fi
        echo "=== INSERT ${ds}.${table} <- ${file} ==="
        t0=${SECONDS}
        if zstd -dc -- "${DATA}/${file}" | client --database "${ds}" --query "INSERT INTO ${table} FORMAT Native"; then
            printf '%s\t%s\n' "${ds}" "$((SECONDS - t0))" >> "${LOAD_STATS}"
            echo "loaded ${ds}.${table}: $(client --database "${ds}" --query "SELECT count() FROM ${table}") rows in $((SECONDS - t0))s"
        else
            echo "LOAD ${ds}.${table} FAILED; dropping"
            client --database "${ds}" --query "DROP TABLE IF EXISTS ${table}" </dev/null 2>/dev/null || true
        fi
    done
}

dataset_fully_loaded() {
    local ds="$1" pair table file cnt
    for pair in ${TABLES[$ds]}; do
        table="${pair%%:*}"; file="${pair##*:}"
        [ -f "${DATA}/${file}" ] || return 1
        table_exists "${ds}" "${table}" || return 1
        cnt="$(client --database "${ds}" --query "SELECT count() FROM ${table}" 2>/dev/null | tr -d '\r')"
        [ -n "${cnt}" ] && [ "${cnt}" != "0" ] || return 1
    done
    return 0
}

load_data() {
    : > "${LOAD_STATS}"
    local ds
    for ds in ${LOAD_DATASETS}; do
        load_one_dataset "${ds}"
    done
}

# Identical timing contract to run-version.sh::run_query
fmt_err() { printf '%s' "$1" | tr '\n\t' '  ' | sed 's/  */ /g; s/^ //' | cut -c1-800; }
run_query() {
    local query="$1" label="${2:-query}" i res rc out="[" skip_rest=0 logged=0
    for i in $(seq 1 "${TRIES}"); do
        if [ "${skip_rest}" = 1 ]; then
            res="null"
        else
            CH_TIMEOUT="${QUERY_TIMEOUT}"
            res=$(printf '%s' "${query}" | client --database "${QDB:-default}" --time --format=Null 2>&1)
            rc=$?
            CH_TIMEOUT=""
            if [ "${rc}" = 124 ] || [ "${rc}" = 137 ]; then
                [ "${logged}" = 0 ] && echo "${label}: FAILED (timeout >${QUERY_TIMEOUT}s); recording null, skipping remaining tries" >&2
                logged=1; skip_rest=1; res="null"
            elif [[ "${res}" =~ ^[0-9]+\.[0-9]+$ ]]; then
                :
            elif ! server_alive; then
                [ "${logged}" = 0 ] && echo "${label}: FAILED (server died); skipping remaining. Last: $(fmt_err "${res}")" >&2
                logged=1; skip_rest=1; res="null"
                # try revive for later queries
                start_server || true
            else
                [ "${logged}" = 0 ] && echo "${label}: FAILED (error): $(fmt_err "${res}")" >&2
                logged=1; res="null"
            fi
        fi
        out+="${res}"
        [ "${i}" -ne "${TRIES}" ] && out+=", "
    done
    echo "${out}]"
}

null_row() {
    local i out="["
    for i in $(seq 1 "${TRIES}"); do out+="null"; [ "${i}" -ne "${TRIES}" ] && out+=", "; done
    echo "${out}]"
}

dump_settings() {
    client --query "SELECT name, value FROM system.settings WHERE name IN ('join_algorithm','max_threads','max_memory_usage') ORDER BY name FORMAT TSV" \
        > "${LOG_DIR}/${VERSION_LABEL}.settings.tsv"
    client --query "SELECT metric, value FROM system.asynchronous_metrics WHERE metric LIKE 'CGroup%' OR metric IN ('NumberOfProcessors','CPUFrequencyMHz_0') ORDER BY metric FORMAT TSV" \
        > "${LOG_DIR}/${VERSION_LABEL}.async_metrics.tsv" || true
    client --query "SELECT getSetting('max_threads')" > "${LOG_DIR}/${VERSION_LABEL}.max_threads.txt"
    client --query "SELECT getSetting('join_algorithm')" > "${LOG_DIR}/${VERSION_LABEL}.join_algorithm.txt"
}

# One-shot EXPLAIN of each dataset's first join-bearing query to prove algorithm.
verify_join_algo() {
    local ds q
    mkdir -p "${LOG_DIR}/explain_${VERSION_LABEL}"
    for ds in ${QUERY_ORDER}; do
        dataset_fully_loaded "${ds}" || continue
        # First query of the file (may or may not join; we also dump a forced join probe).
        q="$(head -1 "${VB}/queries/${ds}.sql" | sed 's/;$//')"
        client --database "${ds}" --query "EXPLAIN PLAN actions=1 ${q}" \
            > "${LOG_DIR}/explain_${VERSION_LABEL}/${ds}_q1.txt" 2>&1 || true
    done
}

run_benchmark() {
    local ds query FIRST=1 qnum=0 row QDB ds_loaded FULLY_LOADED=""
    dump_settings
    verify_join_algo
    for ds in ${QUERY_ORDER}; do
        if dataset_fully_loaded "${ds}"; then FULLY_LOADED+=" ${ds}"
        else echo "=== ${ds}: not fully loaded; skipping ===" >&2; fi
    done
    {
        echo '{'
        echo "    \"version\": \"${VERSION_LABEL}\","
        echo "    \"arm\": \"${ARM}\","
        echo "    \"round\": ${ROUND},"
        echo "    \"actual_version\": \"$(client --query 'SELECT version()' | tr -d '\r')\","
        echo "    \"binary\": \"${BIN}\","
        echo "    \"machine_emulated\": \"c7a.4xlarge\","
        echo "    \"tries\": ${TRIES},"
        echo "    \"result\":"
        echo "    ["
        for ds in ${QUERY_ORDER}; do
            QDB="${ds}"
            case " ${FULLY_LOADED} " in *" ${ds} "*) ds_loaded=1 ;; *) ds_loaded=0 ;; esac
            while IFS= read -r query <&3; do
                [ -z "${query}" ] && continue
                query="${query%;}"
                qnum=$((qnum + 1))
                if [ "${ds_loaded}" = 0 ]; then
                    row="$(null_row)"
                    echo "q${qnum} [${ds}]: SKIPPED (not loaded)" >&2
                else
                    drop_caches
                    row="$(run_query "${query}" "q${qnum} [${ds}]")"
                    echo "q${qnum} [${ds}]: ${row}" >&2
                fi
                [ "${FIRST}" = 0 ] && echo ','
                FIRST=0
                printf '%s' "${row}"
            done 3< "${VB}/queries/${ds}.sql"
        done
        echo
        echo '    ]'
        echo '}'
    } > "${OUT}"
    echo "wrote ${OUT}" >&2
    # Also dump join algorithms seen in query_log for this run.
    client --query "SYSTEM FLUSH LOGS" </dev/null >/dev/null 2>&1 || true
    client --query "
        SELECT
            query_kind,
            count() AS n,
            groupUniqArray(join_algorithm) AS algos
        FROM system.query_log
        WHERE type = 'QueryFinish'
          AND query_kind = 'Select'
          AND event_time > now() - INTERVAL 2 DAY
        GROUP BY query_kind
        FORMAT PrettyCompact
    " > "${LOG_DIR}/${VERSION_LABEL}.query_log_algos.txt" 2>&1 || true
}

case "${PHASE}" in
    load)
        start_server
        load_data
        echo "loaded; leaving server running (pid $(cat "${PIDFILE}"))"
        ;;
    bench)
        server_alive || start_server
        run_benchmark
        stop_server
        ;;
    all|*)
        start_server
        load_data
        run_benchmark
        stop_server
        ;;
esac
