#!/usr/bin/env bash
# Start a test server from the given binary (self-contained under tmp/port_audit/),
# run stateless tests, stop the server.
# Usage: run_stateless.sh <out_log> <jobs> [clickhouse-test filters/args...]
set -u
cd /mnt/ch/ClickHouse
ROOT="$PWD"
BIN="${BIN:-$ROOT/build/reldeb/programs/clickhouse}"
OUT_LOG="${1:?out log}"; shift
JOBS="${1:?jobs}"; shift

DATA="$ROOT/tmp/port_audit/stateless_data"
LOGDIR="$ROOT/tmp/port_audit/stateless_log"
mkdir -p "$LOGDIR"
pkill -9 -f "clickhouse server .*port_audit/stateless_data" 2>/dev/null || true
sleep 1
rm -rf "$DATA"
mkdir -p "$DATA/user_files" "$DATA/tmp"
: > "$LOGDIR/stderr.log"

echo "[driver] BIN=$BIN"
"$BIN" server --config-file "$ROOT/programs/server/config.xml" \
    -- --path "$DATA/" --tmp_path "$DATA/tmp/" \
       --user_files_path "$DATA/user_files/" \
       --logger.stderr "$LOGDIR/stderr.log" --logger.log "$LOGDIR/server.log" \
       > "$LOGDIR/server.stdout" 2>&1 &
SRV_PID=$!

cleanup() {
    echo "[driver] stopping server pid $SRV_PID"
    kill "$SRV_PID" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SRV_PID" 2>/dev/null || true
}
trap cleanup EXIT

UP=0
for i in $(seq 1 60); do
    if "$BIN" client --query "SELECT 1" >/dev/null 2>&1; then UP=1; echo "[driver] server up after ${i}s"; break; fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "[driver] SERVER DIED during startup"; tail -25 "$LOGDIR/server.stdout"; exit 2; fi
    sleep 1
done
[ "$UP" = 1 ] || { echo "[driver] server NOT up after 60s"; tail -25 "$LOGDIR/server.stdout"; exit 2; }
"$BIN" client --query "SELECT 'githash', value FROM system.build_options WHERE name='GIT_HASH'"

echo "[driver] running: clickhouse-test -j $JOBS $*"
python3 tests/clickhouse-test -b "$BIN" -j "$JOBS" "$@" > "$OUT_LOG" 2>&1
RC=$?
echo "[driver] clickhouse-test exit=$RC"
echo "[driver] tail of $OUT_LOG:"; tail -30 "$OUT_LOG"
exit $RC
