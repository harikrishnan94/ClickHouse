#!/usr/bin/env bash
# run_join_differential.sh <binary> <tag> <tcp_port> <http_port> [selector]
#
# One arm of the broad stateless differential: starts a scratch server from
# <binary>, runs clickhouse-test over the selector (default 'join'), writes
# the full log to join_tests_<tag>.log and the normalized failed-test list to
# join_tests_<tag>.failures. The gate rule lives in the driver: candidate
# failures must be a subset of baseline failures.
set -uo pipefail

BINARY="$(readlink -f "$1")"; TAG="$2"; TCP_PORT="$3"; HTTP_PORT="$4"; SELECTOR="${5:-join}"
REPO_ROOT=/mnt/ch/ClickHouse
OUT_DIR="$REPO_ROOT/tmp/chj_probe_parity"
SRV_DIR="$OUT_DIR/join_srv_$TAG"
LOG="$OUT_DIR/join_tests_$TAG.log"
FAILURES="$OUT_DIR/join_tests_$TAG.failures"
PY=/usr/bin/python3

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
    <listen_host>127.0.0.1</listen_host>
    <listen_host>127.0.0.2</listen_host>
    <tcp_port>$TCP_PORT</tcp_port>
    <http_port>$HTTP_PORT</http_port>
    <path>$SRV_DIR/data/</path>
    <tmp_path>$SRV_DIR/data/tmp/</tmp_path>
    <user_files_path>$SRV_DIR/data/user_files/</user_files_path>
    <format_schema_path>$SRV_DIR/data/format_schemas/</format_schema_path>
    <mlock_executable>false</mlock_executable>
    <query_log>
        <database>system</database>
        <table>query_log</table>
        <flush_interval_milliseconds>1000</flush_interval_milliseconds>
    </query_log>
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

SERVER_PID=""
stop_server()
{
    if [ -n "$SERVER_PID" ] && [ -d "/proc/$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null
        for _ in $(seq 1 30); do
            [ ! -d "/proc/$SERVER_PID" ] && break
            sleep 1
        done
        [ -d "/proc/$SERVER_PID" ] && kill -9 "$SERVER_PID" 2>/dev/null
    fi
}
trap stop_server EXIT

(
    cd "$SRV_DIR" || exit 1
    CLICKHOUSE_WATCHDOG_ENABLE=0 nohup "$BINARY" server -C "$SRV_DIR/config.xml" \
        > "$SRV_DIR/logs/server_stdout.log" 2>&1 &
    echo $! > "$SRV_DIR/server.pid"
)
SERVER_PID="$(cat "$SRV_DIR/server.pid")"

ready=0
for _ in $(seq 1 120); do
    [ ! -d "/proc/$SERVER_PID" ] && { echo "FATAL: server died on startup"; tail -20 "$SRV_DIR/logs/clickhouse-server.err.log"; exit 1; }
    if "$BINARY" client --port "$TCP_PORT" -q 'SELECT 1' >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
done
[ "$ready" = 1 ] || { echo "FATAL: server not ready"; exit 1; }

# Binary identity: /proc/<pid>/exe, never the embedded git hash.
exe_sha="$(sha256sum "/proc/$SERVER_PID/exe" | cut -d' ' -f1)"
bin_sha="$(sha256sum "$BINARY" | cut -d' ' -f1)"
[ "$exe_sha" = "$bin_sha" ] || { echo "FATAL: server binary mismatch"; exit 1; }
echo "server pid=$SERVER_PID sha256=$exe_sha selector=$SELECTOR"

tests_tmp="$SRV_DIR/tests_tmp"
mkdir -p "$tests_tmp"
(
    cd "$REPO_ROOT/tests" || exit 1
    CLICKHOUSE_PORT_TCP=$TCP_PORT CLICKHOUSE_PORT_HTTP=$HTTP_PORT \
        $PY ./clickhouse-test -b "$BINARY" --no-random-settings --no-random-merge-tree-settings \
        --jobs 8 --tmp "$tests_tmp" "$SELECTOR"
) > "$LOG" 2>&1
rc=$?

grep -E '^[0-9]+_[a-zA-Z0-9_]+.*\[ FAIL' "$LOG" | sed -E 's/^([0-9]+_[a-zA-Z0-9_]+).*/\1/' | sort -u > "$FAILURES"
echo "clickhouse-test rc=$rc; failures: $(wc -l < "$FAILURES")"
tail -6 "$LOG"
exit 0
