#!/usr/bin/env bash
# D9 -- key8 / key16 in a parallel build. Baseline chooseMethod has no two-level form for these
# two types and returns them unchanged (HashJoin/HashJoin.cpp:418, `default: return type`), so each
# parallel_hash shard owns a private single-level FixedHashMap, twoLevelMapIsUsed() is false, and
# every probe block goes through dispatchBlock (ConcurrentHashJoin.cpp:455-464). UHJ's key8/key16
# are PartitionedFixedHashMap: one shared flat buffer, 256 buckets routing locks rather than owning
# storage, and no probe scatter at all.
#
# Note the asymmetry is probe-side only. UHJ still scatters the *build* side for these types --
# scatterBlockBySlot produces dense_keys whenever the key columns fit in eight bytes per row
# (SlotScatter.cpp:101-123) -- so this is "scatter on both sides versus scatter on the build side
# only, plus bucket locks", not "scatter versus no scatter".
#
# See SPEC_MAPS.md section 3. Everything here is INNER on purpose: a RIGHT variant would traverse
# the map and measure D4 instead.
#
# Usage: ./m_D9.sh            (REPS, THREADS, WANT_ARM overridable in the environment)
set -euo pipefail

# shellcheck source=_maps_common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_maps_common.sh"

maps_take_lock
maps_init D9
trap 'stop_server' EXIT

THREADS="${THREADS:-1 2 4 8 16}"

MODE=""
ALGO=""

do_case() {   # do_case <name> <max_threads> <sql> [extra settings...]
    local name="$1" mt="$2" sql="$3"; shift 3
    case "${MODE}" in
        maptype) capture_maptype "${name}" "${ALGO}" "${mt}" "${SYNTH_DB}" "${sql}" "$@" ;;
        check)   record_result   "${name}" "${ALGO}" "${mt}" "${SYNTH_DB}" "${sql}" "$@" ;;
        run)     run_point       "${name}" "${ALGO}" "${mt}" "${SYNTH_DB}" "${sql}" "$@" ;;
    esac
}

inner_q() { printf 'SELECT count(), sum(r.v) FROM %s AS l INNER JOIN %s AS r ON l.k = r.k' "$1" "$2"; }

cases() {
    local mt
    for mt in ${THREADS}; do
        # --- probe-heavy: 100 M probe rows over a small build side. This is where the
        # --- baseline's per-block dispatchBlock is pure overhead: the key8 map is 6 KB and
        # --- lives in L1, so scattering the block costs more than the lookups it distributes.
        do_case k8_probe  "${mt}" "$(inner_q d9_probe_k8  d9_dim_k8)"
        # key16: 65536 cells, no longer L1-resident, so lookups start to dominate the scatter.
        do_case k16_probe "${mt}" "$(inner_q d9_probe_k16 d9_dim_k16)"

        # --- control: the same 256 values eight bytes wide, with the fixed-map conversion off,
        # --- so both arms use key64 / two_level_key64. D9 and D4 are both switched off here and
        # --- whatever gap remains belongs to the rest of the fork. Subtract it from the rows
        # --- above before attributing anything to D9.
        do_case k64_probe "${mt}" "$(inner_q d9_probe_k64 d9_dim_k64)" \
            --enable_join_fixed_hash_table_conversion=0

        # --- build-heavy: 50 M build rows, k = number % 256, so every block carries all 256 keys
        # --- and therefore has rows for every slot. The shared PartitionedFixedHashMap must take
        # --- all N bucket locks per block while parallel_hash's shards each fill a private
        # --- buffer. With a one-byte key the payload per lock acquisition is as small as it gets.
        do_case k8_build  "${mt}" "$(inner_q d9_probe_k8_small d9_build_k8)"

        # --- as shipped: runtime filters are on by default and are built from exactly these maps.
        do_case k8_probe_rtf "${mt}" "$(inner_q d9_probe_k8 d9_dim_k8)" \
            --enable_join_runtime_filters=1
    done
}

arm() {   # arm <join_algorithm>
    ALGO="$1"
    want_this_arm "${ALGO}" || return 0
    hr; echo "### arm: ${ALGO} (binary $(basename "${CUR_BIN}"))"
    MODE=maptype; cases
    MODE=check;   cases
    MODE=run;     cases
}

## --------------------------------------------------------------------------------------------

start_server baseline
ensure_synth_maps
maps_enable_perfev
arm hash
arm parallel_hash
flush_logs

start_server uhj
arm unified_hash

if [ "${PERF:-0}" = "1" ]; then
    perf_point d9_k8_probe_ph_mem  "${PERF_MEM}"  "${SYNTH_DB}" "$(inner_q d9_probe_k8 d9_dim_k8)" parallel_hash 16
    perf_point d9_k8_probe_uhj_mem "${PERF_MEM}"  "${SYNTH_DB}" "$(inner_q d9_probe_k8 d9_dim_k8)" unified_hash  16
    perf_point d9_k8_build_uhj_core "${PERF_CORE}" "${SYNTH_DB}" "$(inner_q d9_probe_k8_small d9_build_k8)" unified_hash 16
fi

collect_stages
collect_qlog
summarize

cat <<'EOF'

How to read this
  * maptypes.txt: parallel_hash prints one construction line per shard, all saying key8 or key16,
    and the line count is the shard count. unified_hash prints one. That count is the divergence,
    stated directly by the server.
  * At max_threads = 1 both arms are serial and the gap should be about zero. The slope in
    max_threads is D9; a single 16-thread row cannot distinguish "faster" from "scales further".
  * Report throughput per thread as well as wall clock. The failure mode worth finding is a
    shared map that stops scaling at eight threads, which a 16-thread wall-clock table hides.
  * k64_probe must be within noise. If it is not, the k8 and k16 rows are measuring the rest of
    the fork and their gap has to be corrected by it.
  * d9_probe_k8 is 256 distinct one-byte values and compresses extremely well, so a large share
    of the query is decompression both arms pay equally. When the wall-clock gap looks small,
    read instr_per_probe_row from qlog.tsv instead.
EOF

stop_server
trap - EXIT
echo
echo "M_D9_DONE"
