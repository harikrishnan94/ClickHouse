#!/usr/bin/env bash
# Stage 6: stop any prior perf server, hardlink a second data dir, start pre+post.
# Never uses pkill -f; only pidfiles. Does not overwrite clickhouse.pre or clickhouse.base3.
# PRE is clickhouse.base3: the Stage 0 baseline (HEAD fa77c89eb39, BuildID asserted there).
set -euo pipefail

REPO="/mnt/ch/ClickHouse"
RUNDIR="$REPO/tmp/uhj_parity/perf"
POST_BIN="$REPO/build/reldeb/programs/clickhouse"
PRE_BIN="$REPO/build/reldeb/programs/clickhouse.base3"
POST_ROOT="$RUNDIR/chserver"
PRE_ROOT="$RUNDIR/chserver_pre"
POST_PID="$RUNDIR/server_post.pid"
PRE_PID="$RUNDIR/server_pre.pid"
LEGACY_PID="$RUNDIR/server.pid"

POST_TCP=9111
POST_HTTP=8121
PRE_TCP=9112
PRE_HTTP=8122

stop_pidfile() {
  local pf="$1"
  if [ ! -f "$pf" ]; then
    return 0
  fi
  local old
  old=$(cat "$pf" || true)
  if [ -n "${old:-}" ] && kill -0 "$old" 2>/dev/null; then
    echo "Stopping pid=$old from $pf"
    kill -TERM "$old" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "$old" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$old" 2>/dev/null; then
      echo "WARN: pid $old still alive after TERM; sending KILL" >&2
      kill -KILL "$old" 2>/dev/null || true
      sleep 1
    fi
  fi
  rm -f "$pf"
}

write_configs() {
  local tag="$1" tcp="$2" http="$3" root="$4"
  local cfg="$RUNDIR/config_${tag}.xml"
  local users="$RUNDIR/users.xml"
  if [ ! -f "$users" ]; then
    cat > "$users" <<EOF
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
  fi
  mkdir -p "$root/tmp" "$root/user_files"
  cat > "$cfg" <<EOF
<clickhouse>
    <logger>
        <level>warning</level>
        <log>$RUNDIR/server_${tag}.log</log>
        <errorlog>$RUNDIR/server_${tag}.err.log</errorlog>
        <size>100M</size><count>1</count>
    </logger>
    <listen_host>127.0.0.1</listen_host>
    <tcp_port>$tcp</tcp_port>
    <http_port>$http</http_port>
    <path>$root/</path>
    <tmp_path>$root/tmp/</tmp_path>
    <user_files_path>$root/user_files/</user_files_path>
    <users_config>$users</users_config>
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
  echo "$cfg"
}

start_one() {
  local tag="$1" bin="$2" tcp="$3" http="$4" root="$5" pidfile="$6"
  local cfg
  cfg=$(write_configs "$tag" "$tcp" "$http" "$root")
  if ss -ltn 2>/dev/null | grep -qE "127\\.0\\.0\\.1:${tcp}\\b"; then
    echo "ERROR: tcp port $tcp already in use" >&2
    exit 1
  fi
  setsid "$bin" server -C "$cfg" >"$RUNDIR/server_${tag}.stdout" 2>&1 &
  echo $! > "$pidfile"
  echo "Started $tag pid=$(cat "$pidfile") tcp=$tcp http=$http bin=$bin"
  for _ in $(seq 1 120); do
    if "$bin" client --host 127.0.0.1 --port "$tcp" -q "SELECT 1" >/dev/null 2>&1; then
      echo "SERVER_READY tag=$tag port=$tcp pid=$(cat "$pidfile") BuildID=$(file "$bin" | tr ',' '\n' | grep -i buildid | tr -d ' ')"
      return 0
    fi
    if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      echo "SERVER_DIED tag=$tag; see $RUNDIR/server_${tag}.err.log" >&2
      tail -40 "$RUNDIR/server_${tag}.err.log" >&2 || true
      exit 1
    fi
    sleep 1
  done
  echo "SERVER_NOT_READY tag=$tag" >&2
  exit 1
}

echo "=== Stage 6 dual-server bring-up ==="
stop_pidfile "$LEGACY_PID"
stop_pidfile "$POST_PID"
stop_pidfile "$PRE_PID"

# Also stop watchdog children that may linger without pidfile match: check ports.
for p in "$POST_TCP" "$PRE_TCP" "$POST_HTTP" "$PRE_HTTP"; do
  if ss -ltn 2>/dev/null | grep -qE "127\\.0\\.0\\.1:${p}\\b"; then
    echo "ERROR: port $p still listening after pidfile stop" >&2
    ss -ltnp | grep -E ":${p}\\b" || true
    exit 1
  fi
done

if [ ! -x "$POST_BIN" ] || [ ! -x "$PRE_BIN" ]; then
  echo "ERROR: missing binaries" >&2
  exit 1
fi

echo "POST BuildID=$(file "$POST_BIN" | tr ',' '\n' | grep -i buildid | tr -d ' ')"
echo "PRE  BuildID=$(file "$PRE_BIN" | tr ',' '\n' | grep -i buildid | tr -d ' ')"

# Hardlink second data dir from post's existing chserver (page-cache sharing).
if [ -d "$PRE_ROOT" ]; then
  echo "Removing prior hardlink tree $PRE_ROOT"
  rm -rf "$PRE_ROOT"
fi
echo "Hardlinking $POST_ROOT -> $PRE_ROOT"
cp -al "$POST_ROOT" "$PRE_ROOT"
rm -f "$PRE_ROOT/status"
rm -rf "$PRE_ROOT/data/system" "$PRE_ROOT/metadata/system"
# Fresh uuid so the two servers do not share identity metadata.
rm -f "$PRE_ROOT/uuid"
# Drop leftover pid/tmp noise that must not be shared writable.
rm -rf "$PRE_ROOT/tmp"/*
mkdir -p "$PRE_ROOT/tmp" "$PRE_ROOT/user_files"

start_one post "$POST_BIN" "$POST_TCP" "$POST_HTTP" "$POST_ROOT" "$POST_PID"
start_one pre  "$PRE_BIN"  "$PRE_TCP"  "$PRE_HTTP"  "$PRE_ROOT"  "$PRE_PID"

echo "DUAL_SERVERS_READY post_http=$POST_HTTP pre_http=$PRE_HTTP"
