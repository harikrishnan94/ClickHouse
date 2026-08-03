#!/usr/bin/env bash
# Start the measurement server for the uhj perf-attribution mission.
#
# Differs from tmp/uhj_parity/start_server.sh in two ways that the mission needs:
#   - processors_profile_log is configured, so `log_processors_profiles=1` can
#     supply the per-phase (build vs probe) split that Gate G0.5 reconciles.
#   - the data directory is separate, so this mission never races the prior
#     mission's server state.
#
# Never touches :9000 (another task owns it) and refuses to steal a busy port.
set -euo pipefail

CH_BIN="${CH_BIN:-/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse}"
PORT="${UHJ_PORT:-9111}"
HTTP_PORT="${UHJ_HTTP_PORT:-8121}"
RUNDIR="/mnt/ch/ClickHouse/tmp/uhj_parity/perf"
ROOT="$RUNDIR/chserver"
PIDFILE="$RUNDIR/server.pid"

mkdir -p "$ROOT/tmp" "$ROOT/user_files" "$RUNDIR"

if [ -f "$PIDFILE" ]; then
  old=$(cat "$PIDFILE")
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    exe=$(readlink "/proc/$old/exe" 2>/dev/null || true)
    if [[ "$exe" == *clickhouse* ]]; then
      echo "Server already running pid=$old port=$PORT exe=$exe"
      exit 0
    fi
  fi
fi

if ss -ltn 2>/dev/null | grep -qE "127\.0\.0\.1:${PORT}\b"; then
  echo "ERROR: port $PORT already in use; refuse to steal. Set UHJ_PORT differently." >&2
  exit 1
fi

cat > "$RUNDIR/users.xml" <<EOF
<clickhouse>
    <profiles><default>
        <max_memory_usage>0</max_memory_usage>
        <max_memory_usage_for_user>0</max_memory_usage_for_user>
        <log_queries>1</log_queries>
        <log_processors_profiles>1</log_processors_profiles>
        <allow_introspection_functions>1</allow_introspection_functions>
    </default></profiles>
    <users><default>
        <password></password>
        <networks><ip>127.0.0.1</ip><ip>::1</ip></networks>
        <profile>default</profile><quota>default</quota>
        <access_management>1</access_management>
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
    <mark_cache_size>4000000000</mark_cache_size>
    <max_server_memory_usage>0</max_server_memory_usage>
    <query_log>
        <database>system</database><table>query_log</table>
        <engine>ENGINE = MergeTree PARTITION BY event_date ORDER BY event_time</engine>
        <flush_interval_milliseconds>500</flush_interval_milliseconds>
    </query_log>
    <processors_profile_log>
        <database>system</database><table>processors_profile_log</table>
        <engine>ENGINE = MergeTree PARTITION BY event_date ORDER BY event_time</engine>
        <flush_interval_milliseconds>500</flush_interval_milliseconds>
    </processors_profile_log>
    <trace_log>
        <database>system</database><table>trace_log</table>
        <engine>ENGINE = MergeTree PARTITION BY event_date ORDER BY event_time</engine>
        <flush_interval_milliseconds>500</flush_interval_milliseconds>
    </trace_log>
    <query_thread_log remove="1"/>
    <metric_log remove="1"/>
    <asynchronous_metric_log remove="1"/>
    <latency_log remove="1"/>
    <error_log remove="1"/>
</clickhouse>
EOF

setsid "$CH_BIN" server -C "$RUNDIR/config.xml" >"$RUNDIR/server.stdout" 2>&1 &
echo $! > "$PIDFILE"
echo "Started pid=$(cat "$PIDFILE") port=$PORT"

for _ in $(seq 1 90); do
  if "$CH_BIN" client --host 127.0.0.1 --port "$PORT" -q "SELECT 1" >/dev/null 2>&1; then
    echo "SERVER_READY port=$PORT pid=$(cat "$PIDFILE")"
    echo "BINARY_BUILD_ID=$(file "$CH_BIN" | tr ',' '\n' | grep -i buildid | tr -d ' ')"
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
