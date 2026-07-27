#!/usr/bin/env bash
# broken_run_order.sh - DELIBERATELY BROKEN copy of run_order.sh. NOT the real
# harness: this is the selftest fixture for order/SELFTEST.md "Re-proof (c)",
# proving run_order.sh's fail-closed machinery can actually catch a broken
# run. Two TEST-ONLY breaks (grep 'TEST-ONLY BREAK'):
#   1. run_control_count: the inner_all_k control counts a bogus table
#      (order_db.rt_bogus), so the row-count cross-check must trip;
#   2. chj_probe_counter: the engagement counter is pinned to 0, so the
#      engagement gate must trip.
# Never use this for a real gate run; drive it only from the SELFTEST
# re-proof. Everything else is a copy of run_order.sh as of commit
# 91469b6b22e and is NOT kept in sync with it.

set -u

# ============================================================================
# Constants
# ============================================================================
ORDER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRV_DIR="$ORDER_DIR/srv"
LOG_DIR="$ORDER_DIR/logs"
REPO_ROOT="$(cd "$ORDER_DIR/../../.." && pwd)"

TCP_PORT=19310
HTTP_PORT=18310

LT_ROWS=30000000        # left MergeTree table, tag = 0..LT_ROWS-1, physically sorted
LT_KEY_MOD=8000000      # lt.k = tag % LT_KEY_MOD
RT_ROWS=4000000         # right Memory table
RT_KEY_BASE=6000000     # rt.k = intDiv(number,2)*2 + RT_KEY_BASE: evens in [6M,10M), each twice
                        # => matched lt.k: evens in [6M,8M) (1M distinct, ~3 lt rows each, 2 rt dups)
                        # => RIGHT/FULL have unmatched right rows; LEFT/ANTI have unmatched left rows
SCOPE_FILTER="lt.k >= $RT_KEY_BASE"  # scopes RIGHT/FULL output to rows with a real left row
                                     # (unmatched right rows get default lt.k = 0)

# Explicit squash sizes for the powered variants (defaults today, but pinned so
# the oracle does not drift if server defaults change).
SQUASH_ROWS=65409
SQUASH_BYTES=10485760

# The seven ProfileEvents shared by both arms (baseline a05f3ee81ff and phj-ph).
SHARED_EVENTS=(
    ConcurrentHashJoinBuildMicroseconds
    ConcurrentHashJoinBuildDispatchMicroseconds
    ConcurrentHashJoinBuildInsertMicroseconds
    ConcurrentHashJoinBuildMergeMicroseconds
    ConcurrentHashJoinProbeMicroseconds
    ConcurrentHashJoinProbeDispatchMicroseconds
    ConcurrentHashJoinProbeLookupMicroseconds
)
ENGAGEMENT_EVENT=ConcurrentHashJoinProbeMicroseconds  # >0 in query_log <=> parallel_hash probed

# ----------------------------------------------------------------------------
# FUTURE (Units 2-3) contract - keep this block in sync with the mission spec.
# None of these exist in any binary today; everything depending on them
# auto-detects availability and prints a loud SKIPPED line when absent.
# ----------------------------------------------------------------------------
AMAC_EVENTS=(
    ConcurrentHashJoinAmacBuildRows
    ConcurrentHashJoinAmacBuildRingGrowths
    ConcurrentHashJoinAmacProbeRows
)
AMAC_ENGAGEMENT_EVENT=ConcurrentHashJoinAmacProbeRows
AMAC_ENV_VAR=CLICKHOUSE_JOIN_AMAC   # values: 0/off, 1/auto, force; read by server at start
# ----------------------------------------------------------------------------

# ============================================================================
# Arguments
# ============================================================================
EXPECT_FAIL=0
REQUIRE_ENGAGEMENT=0
SKIP_STATELESS=0
KEEP_DATA=0
BINARY=""
for arg in "$@"; do
    case "$arg" in
        --expect-fail) EXPECT_FAIL=1 ;;
        --require-engagement) REQUIRE_ENGAGEMENT=1 ;;
        --skip-stateless) SKIP_STATELESS=1 ;;
        --keep-data) KEEP_DATA=1 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) BINARY="$arg" ;;
    esac
done
if [ -z "$BINARY" ]; then
    echo "usage: run_order.sh <clickhouse-binary> [--expect-fail] [--require-engagement] [--skip-stateless] [--keep-data]" >&2
    exit 2
fi
if [ "$EXPECT_FAIL" = "1" ] && [ "$REQUIRE_ENGAGEMENT" = "1" ]; then
    # --expect-fail is the power check for pre-AMAC scatter binaries, which by
    # definition cannot have AMAC engagement; refuse the contradictory combination.
    echo "--require-engagement is incompatible with --expect-fail" >&2
    exit 2
fi
BINARY="$(readlink -f "$BINARY")"
if [ ! -x "$BINARY" ]; then
    echo "not an executable: $BINARY" >&2
    exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR" "$SRV_DIR"

log() { echo "[run_order] $*"; }

# ============================================================================
# Server lifecycle
# ============================================================================
SERVER_PID=""

client() {
    "$BINARY" client --host 127.0.0.1 --port "$TCP_PORT" "$@"
}

stop_server() {
    # Kill strictly by OUR recorded PID; verify identity via /proc/<pid>/exe
    # inode-independent hash comparison is done once at startup; here we only
    # require that the cmdline still references our config path (PID-reuse guard).
    if [ -n "$SERVER_PID" ] && [ -d "/proc/$SERVER_PID" ]; then
        if tr '\0' ' ' < "/proc/$SERVER_PID/cmdline" 2>/dev/null | grep -qF "$SRV_DIR/config.xml"; then
            kill "$SERVER_PID" 2>/dev/null
            for _ in $(seq 1 30); do
                [ ! -d "/proc/$SERVER_PID" ] && break
                sleep 1
            done
            if [ -d "/proc/$SERVER_PID" ]; then
                kill -9 "$SERVER_PID" 2>/dev/null
            fi
        fi
    fi
    SERVER_PID=""
}
trap stop_server EXIT

write_config() {
    mkdir -p "$SRV_DIR/data" "$SRV_DIR/logs"
    cat > "$SRV_DIR/config.xml" <<EOF
<clickhouse>
    <logger>
        <level>information</level>
        <log>$SRV_DIR/logs/clickhouse-server.log</log>
        <errorlog>$SRV_DIR/logs/clickhouse-server.err.log</errorlog>
        <size>1000M</size>
        <count>2</count>
    </logger>
    <!-- 127.0.0.2 is needed by stateless test 03448, which uses remote('127.0.0.2', ...) -->
    <listen_host>127.0.0.1</listen_host>
    <listen_host>127.0.0.2</listen_host>
    <tcp_port>$TCP_PORT</tcp_port>
    <http_port>$HTTP_PORT</http_port>
    <path>$SRV_DIR/data/</path>
    <tmp_path>$SRV_DIR/data/tmp/</tmp_path>
    <user_files_path>$SRV_DIR/data/user_files/</user_files_path>
    <format_schema_path>$SRV_DIR/data/format_schemas/</format_schema_path>
    <mlock_executable>false</mlock_executable>
    <!-- query_log is REQUIRED: engagement assertions read system.query_log -->
    <query_log>
        <database>system</database>
        <table>query_log</table>
        <flush_interval_milliseconds>1000</flush_interval_milliseconds>
    </query_log>
    <!-- NO keeper/zookeeper section, on purpose -->
    <users>
        <default>
            <password></password>
            <networks><ip>127.0.0.0/8</ip></networks>
            <profile>default</profile>
            <quota>default</quota>
            <access_management>1</access_management>
            <named_collection_control>1</named_collection_control>
        </default>
    </users>
    <profiles><default></default></profiles>
    <quotas><default></default></quotas>
</clickhouse>
EOF
}

start_server() {
    write_config
    # Stop a stale server from a previous run of THIS harness (identified by
    # OUR pidfile + OUR config path in its cmdline; never by pattern).
    if [ -f "$SRV_DIR/server.pid" ]; then
        local old_pid
        old_pid="$(cat "$SRV_DIR/server.pid")"
        if [ -n "$old_pid" ] && [ -d "/proc/$old_pid" ] \
            && tr '\0' ' ' < "/proc/$old_pid/cmdline" 2>/dev/null | grep -qF "$SRV_DIR/config.xml"; then
            log "stopping stale server from previous run (pid=$old_pid)"
            kill "$old_pid" 2>/dev/null
            for _ in $(seq 1 30); do
                [ ! -d "/proc/$old_pid" ] && break
                sleep 1
            done
            [ -d "/proc/$old_pid" ] && kill -9 "$old_pid" 2>/dev/null
        fi
        rm -f "$SRV_DIR/server.pid"
    fi
    # Propagate the FUTURE AMAC diagnostic hook to the server process if the
    # caller set it (no-op on binaries that do not read it).
    if [ -n "${!AMAC_ENV_VAR:-}" ]; then
        export "$AMAC_ENV_VAR"
    fi
    # Start from INSIDE the scratch dir: server-generated preprocessed_configs/
    # land in the server CWD.
    (
        cd "$SRV_DIR" || exit 1
        # Watchdog disabled so $! is the real server PID (see programs/server/Server.cpp,
        # CLICKHOUSE_WATCHDOG_ENABLE). $AMAC_ENV_VAR is passed through from the caller
        # if set (FUTURE Units 2-3 diagnostic hook; a no-op on today's binaries).
        CLICKHOUSE_WATCHDOG_ENABLE=0 nohup "$BINARY" server -C "$SRV_DIR/config.xml" \
            > "$SRV_DIR/logs/server_stdout_$RUN_ID.log" 2>&1 &
        echo $! > "$SRV_DIR/server.pid"
    )
    SERVER_PID="$(cat "$SRV_DIR/server.pid")"
    log "server started, pid=$SERVER_PID (tcp=$TCP_PORT http=$HTTP_PORT), $AMAC_ENV_VAR=${!AMAC_ENV_VAR:-<unset>}"

    local ready=0
    for _ in $(seq 1 120); do
        if [ ! -d "/proc/$SERVER_PID" ]; then
            log "FATAL: server process died during startup; last log lines:"
            tail -20 "$SRV_DIR/logs/clickhouse-server.err.log" 2>/dev/null
            exit 1
        fi
        if [ "$(client -q 'SELECT 1' 2>/dev/null)" = "1" ]; then
            ready=1
            break
        fi
        sleep 1
    done
    if [ "$ready" != "1" ]; then
        log "FATAL: server did not answer SELECT 1 within 120s"
        exit 1
    fi

    # Identify the running server binary by hashing /proc/<pid>/exe (the embedded
    # VERSION_GITHASH is configure-time-stale and MUST NOT be used for identity).
    local expected got
    expected="$(sha256sum "$BINARY" | cut -d' ' -f1)"
    got="$(sha256sum "/proc/$SERVER_PID/exe" | cut -d' ' -f1)"
    if [ "$expected" != "$got" ]; then
        log "FATAL: /proc/$SERVER_PID/exe sha256 ($got) != $BINARY sha256 ($expected)"
        exit 1
    fi
    log "server binary identity verified: sha256=$expected"
}

# ============================================================================
# Data setup
# ============================================================================
data_ok() {
    # NB: rt is a Memory table; its DATA does not survive a server restart even
    # though its metadata does, so counts (not table existence) are the test.
    local parts lt_cnt rt_cnt
    parts="$(client -q "SELECT count() FROM system.parts WHERE database='order_db' AND table='lt' AND active" 2>/dev/null)"
    lt_cnt="$(client -q "SELECT count() FROM order_db.lt" 2>/dev/null)"
    rt_cnt="$(client -q "SELECT count() FROM order_db.rt" 2>/dev/null)"
    if [ "$parts" = "1" ] && [ "$lt_cnt" = "$LT_ROWS" ] && [ "$rt_cnt" = "$RT_ROWS" ]; then
        return 0
    fi
    log "data state: lt=${lt_cnt:-<n/a>} (want $LT_ROWS) in ${parts:-<n/a>} active parts (want 1), rt=${rt_cnt:-<n/a>} (want $RT_ROWS)"
    return 1
}

setup_data() {
    if [ "$KEEP_DATA" = "1" ]; then
        if data_ok; then
            log "reusing existing order_db (--keep-data)"
            return
        fi
        log "--keep-data requested but order_db incomplete or stale; rebuilding"
    fi
    log "creating order_db (lt: $LT_ROWS rows MergeTree, rt: $RT_ROWS rows Memory)"
    client -q "DROP DATABASE IF EXISTS order_db" || exit 1
    client -q "CREATE DATABASE order_db" || exit 1
    client -q "CREATE TABLE order_db.lt (tag UInt64, k UInt64, ks String) ENGINE = MergeTree ORDER BY tag" || exit 1
    client -q "INSERT INTO order_db.lt SELECT number, number % $LT_KEY_MOD, toString(number % $LT_KEY_MOD) FROM numbers($LT_ROWS)" || exit 1
    client -q "OPTIMIZE TABLE order_db.lt FINAL" || exit 1
    client -q "CREATE TABLE order_db.rt (k UInt64, ks String) ENGINE = Memory" || exit 1
    client -q "INSERT INTO order_db.rt SELECT intDiv(number,2)*2 + $RT_KEY_BASE AS k, toString(intDiv(number,2)*2 + $RT_KEY_BASE) FROM numbers($RT_ROWS)" || exit 1

    if ! data_ok; then
        log "FATAL: order_db validation failed after rebuild (need lt=$LT_ROWS rows in 1 active part, rt=$RT_ROWS rows)"
        exit 1
    fi
    log "data ready: lt=$LT_ROWS rows in 1 active part, rt=$RT_ROWS rows"
}

# ============================================================================
# Counter availability auto-detection (single pass over the server image)
# ============================================================================
# The counter names are string literals in any binary that implements them.
# system.events cannot be used for *availability* (zero-valued events are not
# listed there), so grep the running server image /proc/<pid>/exe once.
AMAC_AVAILABLE=0
SHARED_EVENTS_AVAILABLE=0
detect_counters() {
    local grep_args=() ev found
    for ev in "${SHARED_EVENTS[@]}" "${AMAC_EVENTS[@]}" "$AMAC_ENV_VAR"; do
        grep_args+=(-e "$ev")
    done
    found="$(grep -aoF "${grep_args[@]}" "/proc/$SERVER_PID/exe" 2>/dev/null | sort -u)"

    SHARED_EVENTS_AVAILABLE=1
    for ev in "${SHARED_EVENTS[@]}"; do
        if ! grep -qxF "$ev" <<< "$found"; then
            SHARED_EVENTS_AVAILABLE=0
        fi
    done
    if [ "$SHARED_EVENTS_AVAILABLE" = "1" ]; then
        log "shared ProfileEvents (${SHARED_EVENTS[*]}) present in binary; engagement method: query_log profile-events"
    else
        log "WARNING: shared ProfileEvents NOT all present in this binary (both mission arms are supposed to have them; stale build?); engagement method: EXPLAIN fallback"
    fi

    AMAC_AVAILABLE=1
    for ev in "${AMAC_EVENTS[@]}"; do
        if ! grep -qxF "$ev" <<< "$found"; then
            AMAC_AVAILABLE=0
        fi
    done
    local env_hook="absent"
    grep -qxF "$AMAC_ENV_VAR" <<< "$found" && env_hook="present"
    if [ "$AMAC_AVAILABLE" = "1" ]; then
        log "AMAC counters (${AMAC_EVENTS[*]}) present in binary; env hook $AMAC_ENV_VAR: $env_hook"
    else
        log "AMAC-COUNTERS SKIPPED: ${AMAC_EVENTS[*]} not present in this binary (expected before Unit 2; NB availability is inferred by grepping /proc/<pid>/exe for the literal names — a strings-level check that can false-positive or false-negative); env hook $AMAC_ENV_VAR: $env_hook"
    fi
}

# ============================================================================
# Order checks
# ============================================================================
# query_plan_convert_outer_join_to_inner_join is pinned to 0: the RIGHT/FULL
# scope filter (WHERE lt.k >= 6M) rejects the default lt.k = 0 of non-joined
# right rows, so with the default (=1) the planner rewrites RIGHT->inner and
# FULL->left and the scoped checks would not test RIGHT/FULL joins at all
# (verified empirically on a05f3ee81ff; see SELFTEST.md "Post-review fixes").
COMMON_SETTINGS="join_algorithm='parallel_hash', query_plan_join_swap_table=0, query_plan_convert_outer_join_to_inner_join=0, max_bytes_before_external_join=0, max_bytes_ratio_before_external_join=0, join_use_nulls=0"
SQUASH0_SETTINGS="min_joined_block_size_rows=0, min_joined_block_size_bytes=0"
SQUASH_SETTINGS="min_joined_block_size_rows=$SQUASH_ROWS, min_joined_block_size_bytes=$SQUASH_BYTES"

# name|join clause|where clause (or -)
CHECK_CORES=(
    "inner_all_k|INNER JOIN order_db.rt AS rt ON lt.k = rt.k|-"
    "left_all_k|LEFT JOIN order_db.rt AS rt ON lt.k = rt.k|-"
    "left_any_k|ANY LEFT JOIN order_db.rt AS rt ON lt.k = rt.k|-"
    "left_semi_k|SEMI LEFT JOIN order_db.rt AS rt ON lt.k = rt.k|-"
    "left_anti_k|ANTI LEFT JOIN order_db.rt AS rt ON lt.k = rt.k|-"
    "inner_all_ks|INNER JOIN order_db.rt AS rt ON lt.ks = rt.ks|-"
    "right_all_k_scoped|RIGHT JOIN order_db.rt AS rt ON lt.k = rt.k|$SCOPE_FILTER"
    "full_all_k_scoped|FULL JOIN order_db.rt AS rt ON lt.k = rt.k|$SCOPE_FILTER"
)

# Results, indexed by check name
declare -A CHECK_RESULT    # OK | FAIL | ERROR
declare -A CHECK_ROWS      # rows seen by the checker
declare -A CHECK_ENGAGED   # 1 | 0 (parallel_hash actually ran; see fetch_engagement)
declare -A CHECK_THREADS   # 96 | 1
declare -A CHECK_BARE_SQL  # the SELECT without FORMAT, for the EXPLAIN fallback
declare -A CONTROL_ROWS    # expected rows from the hash-join control count
CHECK_NAMES=()

build_sql() {
    # $1 join clause, $2 where clause or '-', $3 select expr, $4 settings, $5 format or ''
    local join="$1" where="$2" select="$3" settings="$4" format="$5"
    local sql="SELECT $select FROM order_db.lt AS lt $join"
    [ "$where" != "-" ] && sql="$sql WHERE $where"
    sql="$sql SETTINGS $settings"
    [ -n "$format" ] && sql="$sql FORMAT $format"
    echo "$sql"
}

run_check() {
    # $1 name, $2 join clause, $3 where, $4 threads, $5 squash settings, $6 extra checker flags
    local name="$1" join="$2" where="$3" threads="$4" squash="$5" checker_flags="$6"
    local lc="chj-order:$name:$RUN_ID"
    local settings="$COMMON_SETTINGS, $squash, max_threads=$threads, log_comment='$lc'"
    local sql
    sql="$(build_sql "$join" "$where" "lt.tag AS tag" "$settings" "Native")"

    local checker_log="$LOG_DIR/${name}.$RUN_ID.checker.log"
    local client_err="$LOG_DIR/${name}.$RUN_ID.client.err"
    client -q "$sql" 2>"$client_err" \
        | python3 "$ORDER_DIR/check_order.py" --quiet $checker_flags > "$checker_log" 2>&1
    local rcs=("${PIPESTATUS[@]}")

    local final stats rows
    final="$(tail -1 "$checker_log")"
    stats="$(grep -m1 '^STATS ' "$checker_log" || echo 'STATS blocks=0 rows=0 violations=0')"
    rows="$(echo "$stats" | sed -n 's/.*rows=\([0-9]*\).*/\1/p')"

    local result
    if [ "${rcs[0]}" != "0" ]; then
        result="ERROR"    # client failed; do NOT count as an order violation
    elif echo "$final" | grep -q '^ORDER-BLOCKS OK'; then
        result="OK"
    elif echo "$final" | grep -q 'violations in'; then
        result="FAIL"
    else
        result="ERROR"    # parse error / empty stream
    fi

    CHECK_NAMES+=("$name")
    CHECK_RESULT[$name]="$result"
    CHECK_ROWS[$name]="$rows"
    CHECK_THREADS[$name]="$threads"
    CHECK_BARE_SQL[$name]="$(build_sql "$join" "$where" "lt.tag AS tag" "$settings" "")"
    log "check $name (T=$threads): $result [$final] rows=$rows"
}

CONTROL_ERRORS=0

run_control_count() {
    # Doer!=grader cross-check: same relation, order-preserving 'hash' algorithm,
    # count only. The checker's row count must match. A failed or non-numeric
    # control is an ERROR (fail-close): it forbids both ORDER OK and
    # POWER-CHECK OK — never silently skip the comparison.
    local core="$1" join="$2" where="$3"
    local settings="join_algorithm='hash', query_plan_join_swap_table=0, query_plan_convert_outer_join_to_inner_join=0, max_bytes_before_external_join=0, max_bytes_ratio_before_external_join=0, join_use_nulls=0, max_threads=32"
    local sql out rc
    # TEST-ONLY BREAK (broken_run_order.sh): bogus table for one control
    [ "$core" = "inner_all_k" ] && join="INNER JOIN order_db.rt_bogus AS rt ON lt.k = rt.k"
    sql="$(build_sql "$join" "$where" "count()" "$settings" "")"
    out="$(client -q "$sql" 2>"$LOG_DIR/control_${core}.$RUN_ID.err")"
    rc=$?
    if [ "$rc" != "0" ] || ! echo "$out" | grep -qE '^[0-9]+$'; then
        CONTROL_ROWS[$core]=""
        CONTROL_ERRORS=$((CONTROL_ERRORS + 1))
        log "CONTROL-ERROR: control count $core failed (rc=$rc, output='$out'; stderr: $LOG_DIR/control_${core}.$RUN_ID.err)"
        return
    fi
    CONTROL_ROWS[$core]="$out"
    log "control count $core (hash join): $out"
}

verify_scoped_join_types() {
    # Guard for the query_plan_convert_outer_join_to_inner_join=0 pin: the
    # scoped RIGHT/FULL checks must actually plan as right/full joins (with
    # the default =1 the scope filter lets the planner rewrite RIGHT->inner
    # and FULL->left, which silently removes RIGHT/FULL coverage). A wrong
    # join type is a CONTROL-ERROR (fail-close, same as a failed control).
    local spec cname want core name join where sql plan got
    for spec in "right_all_k_scoped=right" "full_all_k_scoped=full"; do
        cname="${spec%%=*}"; want="${spec##*=}"
        for core in "${CHECK_CORES[@]}"; do
            IFS='|' read -r name join where <<< "$core"
            [ "$name" = "$cname" ] || continue
            sql="$(build_sql "$join" "$where" "lt.tag AS tag" "$COMMON_SETTINGS, $SQUASH0_SETTINGS, max_threads=96" "")"
            plan="$(client -q "EXPLAIN actions = 1 $sql" 2>"$LOG_DIR/jointype_${name}.$RUN_ID.err")"
            got="$(grep -m1 -o 'Type: [a-zA-Z]*' <<< "$plan")"
            if [ "$got" = "Type: $want" ]; then
                log "join-type verified: $name plans as '$got' ($(grep -m1 -o 'Algorithm: [^|]*' <<< "$plan" | tr -d ' '))"
            else
                CONTROL_ERRORS=$((CONTROL_ERRORS + 1))
                log "CONTROL-ERROR: $name does not plan as 'Type: $want' (got '${got:-<no Type line>}'); the scoped check would not test $want joins"
            fi
        done
    done
}

fetch_engagement() {
    client -q "SYSTEM FLUSH LOGS" >/dev/null 2>&1
    local name lc probe_us plan
    if [ "$SHARED_EVENTS_AVAILABLE" != "1" ]; then
        log "ENGAGEMENT-FALLBACK: shared ProfileEvents absent in this binary; proving engagement via EXPLAIN plan algorithm instead (static plan choice, weaker than runtime counters)"
    fi
    for name in "${CHECK_NAMES[@]}"; do
        if [ "$SHARED_EVENTS_AVAILABLE" = "1" ]; then
            lc="chj-order:$name:$RUN_ID"
            probe_us="$(client -q "SELECT ProfileEvents['$ENGAGEMENT_EVENT'] FROM system.query_log WHERE type = 'QueryFinish' AND log_comment = '$lc' ORDER BY event_time_microseconds DESC LIMIT 1" 2>/dev/null)"
            if [ -n "$probe_us" ] && [ "$probe_us" != "0" ]; then
                CHECK_ENGAGED[$name]=1
            else
                CHECK_ENGAGED[$name]=0
                log "NOT-ENGAGED: check $name has ${ENGAGEMENT_EVENT}=${probe_us:-<no query_log row>} (parallel_hash did not run)"
            fi
        else
            plan="$(client -q "EXPLAIN actions = 1 ${CHECK_BARE_SQL[$name]}" 2>"$LOG_DIR/explain_${name}.$RUN_ID.err")"
            if grep -q "ConcurrentHashJoin" <<< "$plan"; then
                CHECK_ENGAGED[$name]=1
            else
                CHECK_ENGAGED[$name]=0
                log "NOT-ENGAGED: check $name plan algorithm is not ConcurrentHashJoin: $(grep -m1 'Algorithm:' <<< "$plan" | tr -s ' ')"
            fi
        fi
    done
}

fetch_amac_engagement() {
    # Only meaningful when the counters exist (Units 2-3+).
    if [ "$AMAC_AVAILABLE" != "1" ]; then
        log "AMAC-ENGAGEMENT SKIPPED: counters not present in this binary"
        if [ "$REQUIRE_ENGAGEMENT" = "1" ]; then
            log "FATAL: --require-engagement set but AMAC counters are absent"
            AMAC_ENGAGEMENT_FAILED=1
        fi
        return
    fi
    local total=0 name lc v
    for name in "${CHECK_NAMES[@]}"; do
        lc="chj-order:$name:$RUN_ID"
        v="$(client -q "SELECT ProfileEvents['$AMAC_ENGAGEMENT_EVENT'] FROM system.query_log WHERE type = 'QueryFinish' AND log_comment = '$lc' ORDER BY event_time_microseconds DESC LIMIT 1" 2>/dev/null)"
        total=$((total + ${v:-0}))
    done
    if [ "$total" -gt 0 ]; then
        log "AMAC engaged: $AMAC_ENGAGEMENT_EVENT total over checks = $total"
    else
        log "AMAC-ENGAGEMENT NOT-ENGAGED: counters present but $AMAC_ENGAGEMENT_EVENT stayed 0 over all checks"
        if [ "$REQUIRE_ENGAGEMENT" = "1" ]; then
            AMAC_ENGAGEMENT_FAILED=1
        fi
    fi
}
AMAC_ENGAGEMENT_FAILED=0

# ============================================================================
# Stateless tests (normal mode only)
# ============================================================================
STATELESS_RESULT="skipped"
STATELESS_03448_ENGAGED="n/a"
STATELESS_03711_ENGAGED="n/a"
STATELESS_ENGAGEMENT_FAILED=0

chj_probe_counter() {
    # TEST-ONLY BREAK (broken_run_order.sh): pretend the counter never moves
    echo 0
}

run_stateless() {
    # /usr/bin/python3 has jinja2 (needed to render 03711_read_in_order_through_join.sql.j2);
    # the default python3 (linuxbrew) does not.
    local py=/usr/bin/python3
    if ! $py -c 'import jinja2' 2>/dev/null; then
        log "WARNING: $py lacks jinja2; 03711 (.sql.j2) would be skipped by clickhouse-test"
    fi
    local tests_tmp="$SRV_DIR/tests_tmp"
    mkdir -p "$tests_tmp"
    STATELESS_RESULT="pass"
    local t before after delta rc lg
    for t in 03448_analyzer_array_join_alias_in_join_using_bug 03711_read_in_order_through_join; do
        before="$(chj_probe_counter)"
        lg="$LOG_DIR/stateless_${t}.$RUN_ID.log"
        log "running stateless $t x10 (log: $lg)"
        (
            cd "$REPO_ROOT/tests" || exit 1
            CLICKHOUSE_PORT_TCP=$TCP_PORT CLICKHOUSE_PORT_HTTP=$HTTP_PORT \
                $py ./clickhouse-test -b "$BINARY" --no-random-settings --no-random-merge-tree-settings \
                --test-runs 10 --tmp "$tests_tmp" "$t"
        ) > "$lg" 2>&1
        rc=$?
        after="$(chj_probe_counter)"
        delta=$((after - before))
        local engaged="no"
        [ "$delta" -gt 0 ] && engaged="yes"
        # system.events can only prove engagement if this binary has the shared
        # events at all (zero-valued/absent events are indistinguishable there).
        [ "$SHARED_EVENTS_AVAILABLE" != "1" ] && engaged="unknown (shared ProfileEvents absent in this binary)"
        case "$t" in
            03448*) STATELESS_03448_ENGAGED="$engaged (delta ${ENGAGEMENT_EVENT}=$delta)" ;;
            03711*) STATELESS_03711_ENGAGED="$engaged (delta ${ENGAGEMENT_EVENT}=$delta)" ;;
        esac
        # Stateless engagement GATES the verdict (it is not informational):
        # a run whose stateless tests never exercised parallel_hash proves
        # nothing about join output order.
        if [ "$SHARED_EVENTS_AVAILABLE" = "1" ]; then
            if [ "$engaged" != "yes" ]; then
                STATELESS_ENGAGEMENT_FAILED=1
                log "STATELESS-NOT-ENGAGED: $t did not increment $ENGAGEMENT_EVENT (delta=$delta); parallel_hash did not run — ORDER OK is forbidden"
            fi
        else
            log "STATELESS-ENGAGEMENT SKIPPED: shared ProfileEvents absent in this binary; engagement of $t is unknowable via system.events"
            if [ "$REQUIRE_ENGAGEMENT" = "1" ]; then
                STATELESS_ENGAGEMENT_FAILED=1
                log "FATAL: --require-engagement set but stateless engagement is unknowable (shared ProfileEvents absent)"
            fi
        fi
        # clickhouse-test leaves generated/failure files next to the test
        # sources: <t>.gen.sql (jinja2 render) and <t>[.gen].<pid>.stdout/.stderr
        # on failure. They are gitignored, but a stale .gen.sql would be
        # collected as a real test by a later jinja2-less run — remove them.
        rm -f "$REPO_ROOT/tests/queries/0_stateless/$t".gen.sql \
              "$REPO_ROOT/tests/queries/0_stateless/$t".*.stdout \
              "$REPO_ROOT/tests/queries/0_stateless/$t".*.stderr
        if [ "$rc" != "0" ]; then
            STATELESS_RESULT="fail"
            log "stateless $t FAILED (rc=$rc); tail of log:"
            tail -15 "$lg"
        else
            log "stateless $t x10: PASS; parallel_hash engaged during run: $engaged (server-wide $ENGAGEMENT_EVENT delta=$delta)"
        fi
    done
}

# ============================================================================
# Main
# ============================================================================
log "binary: $BINARY"
log "mode: expect_fail=$EXPECT_FAIL require_engagement=$REQUIRE_ENGAGEMENT skip_stateless=$SKIP_STATELESS keep_data=$KEEP_DATA run_id=$RUN_ID"

start_server
detect_counters
setup_data

# Control counts (hash join, the order-preserving reference algorithm)
for core in "${CHECK_CORES[@]}"; do
    IFS='|' read -r name join where <<< "$core"
    run_control_count "$name" "$join" "$where"
done

# The scoped RIGHT/FULL checks must plan as genuine right/full joins
verify_scoped_join_types

# T=96 checks: squash0 = raw join output blocks (mission-specified settings);
# squash = pinned squashing (the variant with demonstrated power on scatter
# binaries -- see SELFTEST.md: raw scatter blocks are single bucket pieces and
# are trivially ordered, squashing concatenates bucket pieces of one input
# block in bucket order and exposes the scatter reordering).
for core in "${CHECK_CORES[@]}"; do
    IFS='|' read -r name join where <<< "$core"
    run_check "$name" "$join" "$where" 96 "$SQUASH0_SETTINGS" ""
    run_check "${name}_squash" "$join" "$where" 96 "$SQUASH_SETTINGS" ""
done

# T=1 global check (single lane => order must hold globally even on scatter)
IFS='|' read -r _ inner_join inner_where <<< "${CHECK_CORES[0]}"
run_check "inner_all_k_t1_global" "$inner_join" "$inner_where" 1 "$SQUASH0_SETTINGS" "--global"

fetch_engagement
fetch_amac_engagement

# Row-count cross-check against the hash-join controls
ROWMISMATCH=0
for name in "${CHECK_NAMES[@]}"; do
    core="${name%_squash}"
    core="${core%_t1_global}"    # inner_all_k_t1_global -> inner_all_k
    expected="${CONTROL_ROWS[$core]:-}"
    got="${CHECK_ROWS[$name]:-}"
    if [ -n "$expected" ] && [ "$got" != "$expected" ]; then
        log "ROW-MISMATCH: check $name saw $got rows, hash-join control says $expected"
        ROWMISMATCH=1
    fi
done

# ============================================================================
# Verdict
# ============================================================================
N_TOTAL=0; N_OK=0; N_FAIL=0; N_ERROR=0; N_NOT_ENGAGED=0
POWER=0
for name in "${CHECK_NAMES[@]}"; do
    N_TOTAL=$((N_TOTAL + 1))
    case "${CHECK_RESULT[$name]}" in
        OK) N_OK=$((N_OK + 1)) ;;
        FAIL) N_FAIL=$((N_FAIL + 1)) ;;
        *) N_ERROR=$((N_ERROR + 1)) ;;
    esac
    [ "${CHECK_ENGAGED[$name]:-0}" != "1" ] && N_NOT_ENGAGED=$((N_NOT_ENGAGED + 1))
    # A check carries expect-fail POWER only if the failure is trustworthy:
    # engaged (parallel_hash really ran), T=96, AND its row count matches the
    # hash-join control (a FAIL over the wrong relation proves nothing).
    if [ "${CHECK_RESULT[$name]}" = "FAIL" ] && [ "${CHECK_ENGAGED[$name]:-0}" = "1" ] && [ "${CHECK_THREADS[$name]}" = "96" ]; then
        core="${name%_squash}"
        expected="${CONTROL_ROWS[$core]:-}"
        if [ -n "$expected" ] && [ "${CHECK_ROWS[$name]:-}" = "$expected" ]; then
            POWER=1
        else
            log "POWER-INELIGIBLE: $name FAILed but rows=${CHECK_ROWS[$name]:-<none>} does not match the control (${expected:-<control failed>})"
        fi
    fi
done

# Control failures (and join-type violations) are errors, never silently skipped
N_ERROR=$((N_ERROR + CONTROL_ERRORS))

T1_RESULT="${CHECK_RESULT[inner_all_k_t1_global]:-ERROR}"
log "summary: total=$N_TOTAL ok=$N_OK fail=$N_FAIL error=$N_ERROR (incl. control_errors=$CONTROL_ERRORS) not_engaged=$N_NOT_ENGAGED row_mismatch=$ROWMISMATCH t1_global=$T1_RESULT"

if [ "$EXPECT_FAIL" = "1" ]; then
    log "stateless portion SKIPPED (--expect-fail mode; failure modes are noisy on scatter binaries)"
    log "T=1 --global result on this binary: $T1_RESULT (expected OK: single lane preserves order)"
    if [ "$POWER" = "1" ] && [ "$N_ERROR" = "0" ] && [ "$ROWMISMATCH" = "0" ]; then
        echo "ORDER POWER-CHECK OK (check fails on this binary, as expected: >=1 engaged row-matched T=96 FAIL, errors=0, row_mismatch=0)"
        exit 0
    else
        echo "ORDER POWER-CHECK BROKEN (power=$POWER errors=$N_ERROR row_mismatch=$ROWMISMATCH — scatter binary passed the check, or the run was invalid)"
        exit 1
    fi
fi

if [ "$SKIP_STATELESS" = "1" ]; then
    STATELESS_RESULT="skipped"
    log "stateless portion skipped (--skip-stateless)"
else
    run_stateless
    log "stateless engagement: 03448=$STATELESS_03448_ENGAGED 03711=$STATELESS_03711_ENGAGED"
fi

FAILED=0
[ "$N_FAIL" != "0" ] && FAILED=1
[ "$N_ERROR" != "0" ] && FAILED=1
[ "$N_NOT_ENGAGED" != "0" ] && FAILED=1
[ "$ROWMISMATCH" != "0" ] && FAILED=1
[ "$STATELESS_RESULT" = "fail" ] && FAILED=1
[ "$STATELESS_ENGAGEMENT_FAILED" != "0" ] && FAILED=1
[ "$AMAC_ENGAGEMENT_FAILED" != "0" ] && FAILED=1

if [ "$FAILED" = "0" ]; then
    echo "ORDER OK ($N_OK/$N_TOTAL checks pass, all engaged parallel_hash, t1_global=$T1_RESULT, stateless=$STATELESS_RESULT, stateless engagement: 03448=$STATELESS_03448_ENGAGED 03711=$STATELESS_03711_ENGAGED)"
    exit 0
else
    echo "ORDER FAIL (ok=$N_OK fail=$N_FAIL error=$N_ERROR not_engaged=$N_NOT_ENGAGED row_mismatch=$ROWMISMATCH stateless=$STATELESS_RESULT stateless_engagement_failed=$STATELESS_ENGAGEMENT_FAILED amac_required_failed=$AMAC_ENGAGEMENT_FAILED of $N_TOTAL checks)"
    exit 1
fi
