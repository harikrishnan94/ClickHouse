#!/usr/bin/env bash
# Start a minimal ClickHouse server for uhj_parity benches (no embedded Keeper).
# Default port 9101. Does NOT touch the process on :9000.
set -euo pipefail

CH_BIN="${CH_BIN:-/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse}"
PORT="${UHJ_PORT:-9101}"
HTTP_PORT="${UHJ_HTTP_PORT:-8111}"
ROOT="/mnt/ch/ClickHouse/tmp/uhj_parity/chserver"
PIDFILE="/mnt/ch/ClickHouse/tmp/uhj_parity/server.pid"
RUNDIR="/mnt/ch/ClickHouse/tmp/uhj_parity"

mkdir -p "$ROOT/tmp" "$ROOT/user_files" "$RUNDIR"

if [ -f "$PIDFILE" ]; then
  old=$(cat "$PIDFILE")
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    exe=$(readlink "/proc/$old/exe" 2>/dev/null || true)
    if [[ "$exe" == *clickhouse* ]]; then
      echo "Server already running pid=$old port=$PORT"
      exit 0
    fi
  fi
fi

if ss -ltnp 2>/dev/null | rg -q ":${PORT}\\b"; then
  echo "ERROR: port $PORT already in use; refuse to steal. Set UHJ_PORT differently." >&2
  exit 1
fi

cat > "$RUNDIR/users.xml" <<EOF
<clickhouse>
    <profiles><default>
        <max_memory_usage>0</max_memory_usage>
        <log_queries>1</log_queries>
        <allow_introspection_functions>1</allow_introspection_functions>
    </default></profiles>
    <users><default>
        <password></password>
        <networks><ip>127.0.0.1</ip><ip>::1</ip></networks>
        <profile>default</profile><quota>default</quota>
    </default></users>
    <quotas><default></default></quotas>
</clickhouse>
EOF

cat > "$RUNDIR/config.xml" <<EOF
<clickhouse>
    <logger>
        <level>warning</level>
        <log>$RUNDIR/server.log</log>
        <errorlog>$RUNDIR/server.err.log</errorlog>
        <size>100M</size><count>1</count>
    </logger>
    <listen_host>127.0.0.1</listen_host>
    <tcp_port>$PORT</tcp_port>
    <http_port>$HTTP_PORT</http_port>
    <path>$ROOT/</path>
    <tmp_path>$ROOT/tmp/</tmp_path>
    <user_files_path>$ROOT/user_files/</user_files_path>
    <users_config>$RUNDIR/users.xml</users_config>
    <default_profile>default</default_profile>
    <default_database>default</default_database>
    <mark_cache_size>1000000000</mark_cache_size>
    <query_log>
        <database>system</database><table>query_log</table>
        <engine>ENGINE = MergeTree PARTITION BY event_date ORDER BY event_time</engine>
        <flush_interval_milliseconds>750</flush_interval_milliseconds>
    </query_log>
    <trace_log>
        <database>system</database><table>trace_log</table>
        <engine>ENGINE = MergeTree PARTITION BY event_date ORDER BY event_time</engine>
        <flush_interval_milliseconds>750</flush_interval_milliseconds>
    </trace_log>
</clickhouse>
EOF

setsid "$CH_BIN" server -C "$RUNDIR/config.xml" \
  >"$RUNDIR/server.stdout" 2>&1 &
echo $! > "$PIDFILE"
echo "Started pid=$(cat "$PIDFILE") port=$PORT"

for _ in $(seq 1 60); do
  if "$CH_BIN" client --host 127.0.0.1 --port "$PORT" -q "SELECT 1" >/dev/null 2>&1; then
    echo "SERVER_READY port=$PORT"
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "SERVER_DIED; see $RUNDIR/server.err.log" >&2
    tail -40 "$RUNDIR/server.err.log" >&2 || true
    exit 1
  fi
  sleep 1
done
echo "SERVER_NOT_READY" >&2
exit 1
