#!/usr/bin/env bash
# run_parity.sh <baseline_bin> <candidate_bin> [--require-engagement] [--limit N]
#               [--allow-identical] [--acknowledge-matched-errors N]
#               [--cases-override FILE]
#
# Parity gate for the AMAC + order-preserving probe mission (Unit 1).
#   1. Generates the deterministic query matrix (parity_gen.py).
#   2. Starts one server per binary (baseline tcp 19101 / http 18101,
#      candidate tcp 19201 / http 18201) in parity/srv_base, parity/srv_cand;
#      verifies sha256(/proc/<pid>/exe) == sha256(<given binary>) — the
#      embedded VERSION_GITHASH is configure-time-stale and MUST NOT be used
#      for identity.
#   3. Runs every case on both arms (batched multiquery, per-case outputs via
#      INTO OUTFILE), byte-diffs the ORDER BY ALL TSVs, cross-checks the
#      count/cityHash64 checksums, and writes parity/logs/<id>.divergence.txt
#      for EVERY divergence (does not stop at the first).
#   4. Informational audit: per-family, did parallel_hash actually engage
#      (shared ConcurrentHashJoin* ProfileEvents) on each arm.
#   5. AMAC force pass: if the (future, Units 2-3) engagement counters are
#      present in the candidate binary, restarts the candidate with
#      CLICKHOUSE_JOIN_AMAC=force and asserts the counters via
#      system.query_log; otherwise prints a loud SKIPPED line.
#      --require-engagement turns absence into failure.
#
# Final line (machine-checkable):
#   PARITY OK (N cases: C compared, M matched-error, E failed; F families,
#              K kind-strictness combos, force-pass: engaged X/Y|SKIPPED[, identical-binaries])
#   PARITY FAIL (D divergences, G gate failure(s); N cases: C compared,
#              M matched-error, E failed; see parity/logs/[, identical-binaries])
# Exit code 0 only on OK. Engagement failures under force count as
# divergences (a divergence file is written per failing family).
#
# Cases where BOTH arms raise the IDENTICAL genuine server exception
# (signature: Code preserved verbatim, host:port clause removed, only
# genuinely varying tokens — addresses / row counts — collapsed) are
# parity-preserving "matched errors": reported with a loud WARNING and a
# parity/logs/<id>.matched-error.txt file. They are budgeted: more than 4, or
# more than 2% of cases, fails the gate unless explicitly acknowledged with
# --acknowledge-matched-errors N. Connection failures, timeouts, and harness
# errors are NEVER matched errors. At least 90% of cases must produce real
# comparisons or the gate fails (no vacuous passes). An exception on ONE arm
# only, or differing exceptions, is a divergence and fails the gate.
#
# sha256(baseline) == sha256(candidate) is FATAL unless --allow-identical is
# given (self-test mode); the final line then carries an 'identical-binaries'
# marker.
#
# --cases-override FILE is a self-test seam: skip generation and use FILE as
# the case matrix (e.g. a doctored COPY of cases.jsonl). Never use it for a
# real gate run.
#
# Servers are fully managed within this invocation: start -> poll SELECT 1 ->
# use -> stop by PID. PIDs are only killed after verifying /proc/<pid>/cwd is
# our own server directory; if that check fails the script aborts (fail-close,
# pid file kept) instead of continuing next to a possibly-live foreign server.

set -uo pipefail

# pwd -P: canonical physical path. /proc/<pid>/cwd is kernel-resolved, so any
# symlinked invocation path (e.g. /home/ubuntu/ClickHouse -> /mnt/ch/ClickHouse)
# would otherwise break the stop_server ownership check.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

BASE_TCP=19101; BASE_HTTP=18101
CAND_TCP=19201; CAND_HTTP=18201

usage() { echo "usage: $0 <baseline_bin> <candidate_bin> [--require-engagement] [--limit N]" \
               "[--allow-identical] [--acknowledge-matched-errors N] [--cases-override FILE]" >&2; exit 2; }

[ $# -ge 2 ] || usage
BASE_BIN=$(readlink -f "$1"); CAND_BIN=$(readlink -f "$2"); shift 2
REQUIRE_ENGAGEMENT=0; LIMIT=""; ALLOW_IDENTICAL=0; ACK_MATCHED=""; CASES_OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --require-engagement) REQUIRE_ENGAGEMENT=1; shift ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --allow-identical) ALLOW_IDENTICAL=1; shift ;;
        --acknowledge-matched-errors) ACK_MATCHED="$2"; shift 2 ;;
        --cases-override) CASES_OVERRIDE="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[ -x "$BASE_BIN" ] || { echo "FATAL: baseline binary not executable: $BASE_BIN" >&2; exit 2; }
[ -x "$CAND_BIN" ] || { echo "FATAL: candidate binary not executable: $CAND_BIN" >&2; exit 2; }

LIMIT_ARGS=()
[ -n "$LIMIT" ] && LIMIT_ARGS=(--limit "$LIMIT")

CASES="$SCRIPT_DIR/cases.jsonl"
LOGDIR="$SCRIPT_DIR/logs"
SRV_BASE="$SCRIPT_DIR/srv_base"
SRV_CAND="$SCRIPT_DIR/srv_cand"
BASE_CLIENT="$BASE_BIN client --host 127.0.0.1 --port $BASE_TCP"
CAND_CLIENT="$CAND_BIN client --host 127.0.0.1 --port $CAND_TCP"

# --- server management ------------------------------------------------------

stop_server() { # $1 = server dir; returns 1 (pid file KEPT) if the pid is not ours
    local dir=$1 pid cwd
    [ -f "$dir/pid" ] || return 0
    pid=$(cat "$dir/pid")
    if kill -0 "$pid" 2>/dev/null; then
        cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
        if [ "$cwd" = "$(readlink -f "$dir")" ]; then
            kill "$pid" 2>/dev/null
            for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
            kill -0 "$pid" 2>/dev/null && { kill -9 "$pid" 2>/dev/null; sleep 1; }
        else
            # Fail-close: do NOT delete the pid file and do NOT let the caller
            # continue (it would rm -rf the data dirs of a possibly-live
            # server). The caller must abort.
            echo "FATAL: pid $pid cwd '$cwd' is not '$dir'; refusing to kill, keeping $dir/pid;" \
                 "investigate manually" >&2
            return 1
        fi
    fi
    rm -f "$dir/pid"
}

start_server() { # $1 dir, $2 bin, $3 tcp, $4 http, $5 optional VAR=val env
    local dir=$1 bin=$2 tcp=$3 http=$4 extra_env=${5:-} pid status_pid ready=0
    mkdir -p "$dir"
    sed -e "s|@TCP_PORT@|$tcp|g" -e "s|@HTTP_PORT@|$http|g" -e "s|@PATH@|$dir|g" \
        "$SCRIPT_DIR/minimal_config.xml.in" > "$dir/config.xml"
    # exec chain keeps $! == the server pid; the watchdog is disabled so the
    # server is a single process (with it on, the started pid becomes a
    # watchdog parent and pid bookkeeping breaks — observed in self-test).
    ( cd "$dir" && exec env CLICKHOUSE_WATCHDOG_ENABLE=0 $extra_env \
        "$bin" server --config-file="$dir/config.xml" > "$dir/server.stdout" 2>&1 ) &
    pid=$!
    echo "$pid" > "$dir/pid"
    for _ in $(seq 1 240); do
        kill -0 "$pid" 2>/dev/null || { echo "FATAL: server in $dir died at startup:" >&2;
            tail -30 "$dir/server.stdout" "$dir/clickhouse-server.err.log" 2>/dev/null >&2; return 1; }
        "$bin" client --host 127.0.0.1 --port "$tcp" -q "SELECT 1" >/dev/null 2>&1 && { ready=1; break; }
        sleep 0.5
    done
    if [ "$ready" -ne 1 ]; then
        echo "FATAL: server in $dir did not become ready in 120s" >&2
        return 1
    fi
    # Cross-check against the pid the server wrote itself (authoritative).
    status_pid=$(sed -n 's/^PID: //p' "$dir/data/status" 2>/dev/null | head -1)
    if [ -z "$status_pid" ] || [ "$status_pid" != "$pid" ]; then
        echo "FATAL: launched pid $pid != server status pid '${status_pid:-<missing>}' in $dir" >&2
        return 1
    fi
}

verify_exe() { # $1 dir, $2 expected sha256 of the binary file
    local dir=$1 want=$2 pid got
    pid=$(cat "$dir/pid")
    got=$(sha256sum "/proc/$pid/exe" | awk '{print $1}')
    if [ "$got" != "$want" ]; then
        echo "FATAL: /proc/$pid/exe sha256=$got != given binary sha256=$want (dir $dir)" >&2
        return 1
    fi
    echo "verified: server pid $pid in $dir runs the given binary (sha256 $got)"
}

# Background driver PIDs (phase 1); killed by the trap so an interrupted run
# does not leave drivers grinding against stopped servers.
DRIVER_PIDS=()

cleanup() {
    local rc=0 p
    for p in "${DRIVER_PIDS[@]}"; do
        kill "$p" 2>/dev/null || true
        wait "$p" 2>/dev/null || true
    done
    DRIVER_PIDS=()
    stop_server "$SRV_BASE" || rc=1
    stop_server "$SRV_CAND" || rc=1
    if [ "$rc" -ne 0 ]; then
        echo "FATAL: cleanup refused to stop a foreign pid (fail-close); pid file(s) kept" >&2
        exit 3
    fi
}
trap cleanup EXIT

# --- phase 0: generate matrix, clean state ----------------------------------

echo "== phase 0: generation and server startup =="
# Stale servers from a crashed run. Fail-close: a refused stop aborts BEFORE
# the rm -rf below can delete the data dirs of a possibly-live server.
stop_server "$SRV_BASE" || exit 3
stop_server "$SRV_CAND" || exit 3
rm -rf "$SRV_BASE" "$SRV_CAND" "$LOGDIR"
mkdir -p "$LOGDIR"

if [ -n "$CASES_OVERRIDE" ]; then
    CASES=$(readlink -f "$CASES_OVERRIDE")
    [ -s "$CASES" ] || { echo "FATAL: --cases-override $CASES_OVERRIDE missing or empty" >&2; exit 2; }
    echo "NOTE: cases override in effect ($CASES) — self-test seam, generation skipped"
else
    python3 "$SCRIPT_DIR/parity_gen.py" --out "$CASES" --stats 2>&1 | sed 's/^/gen: /' || exit 2
fi

BASE_SHA=$(sha256sum "$BASE_BIN" | awk '{print $1}')
CAND_SHA=$(sha256sum "$CAND_BIN" | awk '{print $1}')
echo "baseline:  $BASE_BIN (sha256 $BASE_SHA)"
echo "candidate: $CAND_BIN (sha256 $CAND_SHA)"

IDENT_MARK=""
if [ "$BASE_SHA" = "$CAND_SHA" ]; then
    if [ "$ALLOW_IDENTICAL" -ne 1 ]; then
        echo "FATAL: baseline and candidate are the SAME binary (sha256 $BASE_SHA)." \
             "A parity gate over identical binaries is vacuous; pass --allow-identical" \
             "only for harness self-tests." >&2
        exit 2
    fi
    IDENT_MARK=", identical-binaries"
    echo "NOTE: identical binaries on both arms (--allow-identical): harness self-test mode"
fi

start_server "$SRV_BASE" "$BASE_BIN" "$BASE_TCP" "$BASE_HTTP" || exit 1
start_server "$SRV_CAND" "$CAND_BIN" "$CAND_TCP" "$CAND_HTTP" || exit 1
verify_exe "$SRV_BASE" "$BASE_SHA" || exit 1
verify_exe "$SRV_CAND" "$CAND_SHA" || exit 1

# --- phase 1: run all cases on both arms in parallel ------------------------

echo "== phase 1: executing cases on both arms =="
python3 "$SCRIPT_DIR/parity_driver.py" run --cases "$CASES" --client "$BASE_CLIENT" \
    --outdir "$SRV_BASE/out" "${LIMIT_ARGS[@]}" > "$LOGDIR/run_base.log" 2>&1 &
RUN_BASE_PID=$!
python3 "$SCRIPT_DIR/parity_driver.py" run --cases "$CASES" --client "$CAND_CLIENT" \
    --outdir "$SRV_CAND/out" "${LIMIT_ARGS[@]}" > "$LOGDIR/run_cand.log" 2>&1 &
RUN_CAND_PID=$!
DRIVER_PIDS=("$RUN_BASE_PID" "$RUN_CAND_PID")   # cleanup trap kills these on abort
wait "$RUN_BASE_PID"; RC_BASE=$?
wait "$RUN_CAND_PID"; RC_CAND=$?
DRIVER_PIDS=()
grep '^RUN ' "$LOGDIR/run_base.log" | sed 's/^/baseline /' || true
grep '^RUN ' "$LOGDIR/run_cand.log" | sed 's/^/candidate /' || true
if [ "$RC_BASE" -ne 0 ] || [ "$RC_CAND" -ne 0 ]; then
    echo "FATAL: run phase failed (baseline rc=$RC_BASE, candidate rc=$RC_CAND); see $LOGDIR/run_*.log" >&2
    tail -15 "$LOGDIR/run_base.log" "$LOGDIR/run_cand.log" >&2
    echo "PARITY FAIL (run phase failed, see parity/logs/)"
    exit 1
fi

# --- phase 2: compare --------------------------------------------------------

echo "== phase 2: comparing arms =="
ACK_ARGS=()
[ -n "$ACK_MATCHED" ] && ACK_ARGS=(--acknowledge-matched-errors "$ACK_MATCHED")
python3 "$SCRIPT_DIR/parity_driver.py" compare --cases "$CASES" \
    --base-dir "$SRV_BASE/out" --cand-dir "$SRV_CAND/out" --logdir "$LOGDIR" \
    --base-label "baseline $BASE_BIN" --cand-label "candidate $CAND_BIN" \
    --base-client "$BASE_CLIENT" --cand-client "$CAND_CLIENT" \
    "${ACK_ARGS[@]}" "${LIMIT_ARGS[@]}" > "$LOGDIR/compare.log" 2>&1
RC_CMP=$?
grep '^DIVERGENCE ' "$LOGDIR/compare.log" || true
COMPARE_JSON=$(grep '^COMPARE ' "$LOGDIR/compare.log" | tail -1 | sed 's/^COMPARE //')
if [ "$RC_CMP" -ne 0 ] || [ -z "$COMPARE_JSON" ]; then
    echo "FATAL: compare phase failed; see $LOGDIR/compare.log" >&2
    tail -15 "$LOGDIR/compare.log" >&2
    echo "PARITY FAIL (compare phase failed, see parity/logs/)"
    exit 1
fi
# GATE_FAILURES is ' | '-joined free text and must stay the LAST read field
# ('-' when empty).
read -r N_CASES N_COMPARED N_FAILED N_FAMILIES N_COMBOS N_DIVERGENCES N_MATCHED_ERRORS GATE_FAILURES <<EOF
$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['cases'], d['compared'], d['failed'], d['families'], d['combos'], d['divergences'], d['matched_errors'], ' | '.join(d['gate_failures']) or '-')" "$COMPARE_JSON")
EOF
echo "compared: $N_CASES cases ($N_COMPARED compared, $N_MATCHED_ERRORS matched-error, $N_FAILED failed)," \
     "$N_FAMILIES families, $N_COMBOS combos, $N_DIVERGENCES divergences"
if [ "$N_MATCHED_ERRORS" -gt 0 ]; then
    echo "WARNING: $N_MATCHED_ERRORS case(s) raised IDENTICAL exceptions on both arms" \
         "(pre-existing product behavior, parity-preserving; see parity/logs/*.matched-error.txt):"
    grep '^MATCHED-ERROR ' "$LOGDIR/compare.log" | sed 's/^/  /'
fi
N_GATE_FAILURES=0
if [ "$GATE_FAILURES" != "-" ]; then
    N_GATE_FAILURES=$(python3 -c "import json,sys; print(len(json.loads(sys.argv[1])['gate_failures']))" "$COMPARE_JSON")
    echo "GATE FAILURE(S): $GATE_FAILURES"
fi

# --- phase 3: informational parallel_hash engagement audit -------------------

echo "== phase 3: parallel_hash engagement audit (informational) =="
SHARED_PROBE_EVENT=$(python3 "$SCRIPT_DIR/parity_gen.py" --print-contract \
    | sed -n 's/^SHARED_EVENT=//p' | grep -E 'ProbeMicroseconds$' | head -1)
for ARM in base cand; do
    if [ "$ARM" = base ]; then CL="$BASE_CLIENT"; BIN="$BASE_BIN"; else CL="$CAND_CLIENT"; BIN="$CAND_BIN"; fi
    if ! LC_ALL=C grep -a -m1 -q -F "$SHARED_PROBE_EVENT" "$BIN"; then
        echo "audit[$ARM]: SKIPPED (shared event $SHARED_PROBE_EVENT absent from binary)"
        continue
    fi
    python3 "$SCRIPT_DIR/parity_driver.py" audit --cases "$CASES" --client "$CL" \
        --label "$ARM" "${LIMIT_ARGS[@]}" > "$LOGDIR/audit_$ARM.log" 2>&1
    AUD=$(grep '^AUDIT ' "$LOGDIR/audit_$ARM.log" | tail -1 | sed 's/^AUDIT //')
    if [ -n "$AUD" ]; then
        python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(f\"audit[$ARM]: parallel_hash engaged in {len(d['parallel_hash_engaged_families'])}/{d['total_families']} families: {','.join(d['parallel_hash_engaged_families'])}\")" "$AUD"
    else
        echo "audit[$ARM]: FAILED (informational); see $LOGDIR/audit_$ARM.log"
    fi
done

# --- phase 4: AMAC force pass (auto-detected; Units 2-3 contract) ------------

echo "== phase 4: AMAC force pass =="
AMAC_ENV_VAR=$(python3 "$SCRIPT_DIR/parity_gen.py" --print-contract | sed -n 's/^AMAC_ENV_VAR=//p')
# Detection requires only the ASSERTED counters; informational ones (e.g.
# RingGrowths) may be absent without disabling the force pass.
AMAC_ASSERT_EVENTS=$(python3 "$SCRIPT_DIR/parity_gen.py" --print-contract | sed -n 's/^AMAC_ASSERT_EVENT=//p')
COUNTERS_PRESENT=1
for EV in $AMAC_ASSERT_EVENTS; do
    if ! LC_ALL=C grep -a -m1 -q -F "$EV" "$CAND_BIN"; then
        COUNTERS_PRESENT=0
        echo "asserted counter '$EV' not found in candidate binary"
        break
    fi
done

FORCE_PASS="SKIPPED"
if [ "$COUNTERS_PRESENT" -eq 1 ]; then
    echo "AMAC asserted counters present in candidate binary; restarting candidate with $AMAC_ENV_VAR=force"
    stop_server "$SRV_CAND" || exit 3
    start_server "$SRV_CAND" "$CAND_BIN" "$CAND_TCP" "$CAND_HTTP" "$AMAC_ENV_VAR=force" || exit 1
    verify_exe "$SRV_CAND" "$CAND_SHA" || exit 1
    python3 "$SCRIPT_DIR/parity_driver.py" engage --cases "$CASES" --client "$CAND_CLIENT" \
        --logdir "$LOGDIR" "${LIMIT_ARGS[@]}" > "$LOGDIR/engage.log" 2>&1
    ENG=$(grep '^ENGAGE ' "$LOGDIR/engage.log" | tail -1 | sed 's/^ENGAGE //')
    if [ -z "$ENG" ]; then
        echo "FATAL: engage phase produced no summary; see $LOGDIR/engage.log" >&2
        tail -15 "$LOGDIR/engage.log" >&2
        echo "PARITY FAIL (engage phase failed, see parity/logs/)"
        exit 1
    fi
    read -r ENG_X ENG_Y <<EOF
$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['engaged'], d['total'])" "$ENG")
EOF
    FORCE_PASS="engaged $ENG_X/$ENG_Y"
    echo "AMAC-FORCE PASS: $FORCE_PASS"
    if [ "$ENG_X" -ne "$ENG_Y" ]; then
        # per-family divergence files were written by the engage driver
        N_DIVERGENCES=$((N_DIVERGENCES + ENG_Y - ENG_X))
    fi
else
    echo "AMAC-FORCE PASS: SKIPPED (counters absent)"
    if [ "$REQUIRE_ENGAGEMENT" -eq 1 ]; then
        {
            echo "CASE engagement-required"
            echo "STATUS: amac-counters-absent"
            echo "--require-engagement was given but the candidate binary lacks the"
            echo "asserted AMAC engagement counters: $AMAC_ASSERT_EVENTS"
        } > "$LOGDIR/engagement-required.divergence.txt"
        N_DIVERGENCES=$((N_DIVERGENCES + 1))
    fi
fi

# --- verdict ------------------------------------------------------------------

COUNTS="$N_CASES cases: $N_COMPARED compared, $N_MATCHED_ERRORS matched-error, $N_FAILED failed"
if [ "$N_DIVERGENCES" -eq 0 ] && [ "$N_GATE_FAILURES" -eq 0 ]; then
    echo "PARITY OK ($COUNTS; $N_FAMILIES families, $N_COMBOS kind-strictness combos, force-pass: $FORCE_PASS$IDENT_MARK)"
    exit 0
else
    echo "PARITY FAIL ($N_DIVERGENCES divergences, $N_GATE_FAILURES gate failure(s); $COUNTS; see parity/logs/$IDENT_MARK)"
    exit 1
fi
