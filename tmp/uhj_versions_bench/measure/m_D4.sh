#!/usr/bin/env bash
# D4 -- every fixed / direct-addressed map in UHJ is a PartitionedFixedHashMap, whose
# FixedRangeStorage constructor calls disableMinMaxOptimization() permanently. A full traversal
# then costs the whole key range instead of the observed [min, max] span.
#
# See SPEC_MAPS.md section 1. In short, there are two penalties and they scale differently:
#   leading  -- firstPopulatedCell()'s linear search from cell 0, paid once per non-joined stream,
#               including the streams that own no iteration bucket and return immediately;
#   trailing -- the walk from the last real key to NUM_CELLS, paid once by the owning stream.
# The `top` / `bot` table pairs separate them, and the max_threads sweep is what distinguishes
# them: only the leading penalty multiplies by stream count.
#
# Read stages.tsv before timings.tsv. The whole effect is a few hundred microseconds of cell
# walking, invisible in wall clock next to a probe side but the entire content of the non-joined
# processor's elapsed_us.
#
# Usage: ./m_D4.sh            (REPS, PERF, WANT_ARM overridable in the environment)
set -euo pipefail

# shellcheck source=_maps_common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_maps_common.sh"

maps_take_lock
maps_init D4
trap 'stop_server' EXIT

THREADS="${THREADS:-1 4 16}"

## --------------------------------------------------------------------------------------------
## Case matrix
##
## `nomatch` probes are four rows that match nothing, so on a RIGHT join every build row is
## non-joined and the traversal is the entire query. Against the two `_full` controls no
## UInt8/UInt16 value can miss, which is what makes them controls.
##
## sum(r.v) is in every select list on purpose: it keeps a right-side column in
## sample_block_with_columns_to_add, without which the rerange case below returns early and
## measures nothing.
## --------------------------------------------------------------------------------------------

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

# join_q <probe table> <RIGHT|FULL|INNER> <build table>
join_q() {
    printf 'SELECT count(), sum(r.v) FROM %s AS l %s JOIN %s AS r ON l.k = r.k' "$1" "$2" "$3"
}

cases() {
    local mt
    for mt in ${THREADS}; do
        # --- the traversal matrix, RIGHT JOIN, four-row non-matching probe -------------------
        do_case k16_top64_right   "${mt}" "$(join_q d4_probe_k16_nomatch RIGHT d4_dim_k16_top64)"
        do_case k16_bot64_right   "${mt}" "$(join_q d4_probe_k16_nomatch RIGHT d4_dim_k16_bot64)"
        do_case k16_top4096_right "${mt}" "$(join_q d4_probe_k16_nomatch RIGHT d4_dim_k16_top4096)"
        do_case k16_full_right    "${mt}" "$(join_q d4_probe_k16_nomatch RIGHT d4_dim_k16_full)"
        do_case k8_top16_right    "${mt}" "$(join_q d4_probe_k8_nomatch  RIGHT d4_dim_k8_top16)"
        do_case k8_full_right     "${mt}" "$(join_q d4_probe_k8_nomatch  RIGHT d4_dim_k8_full)"
        do_case u64_r257_right    "${mt}" "$(join_q d4_probe_u64_nomatch RIGHT d4_dim_u64_r257)"
        do_case u64_r65k_right    "${mt}" "$(join_q d4_probe_u64_nomatch RIGHT d4_dim_u64_r65k)"
        do_case u64_r131k_right   "${mt}" "$(join_q d4_probe_u64_nomatch RIGHT d4_dim_u64_r131k)"
        do_case u64_r262k_right   "${mt}" "$(join_q d4_probe_u64_nomatch RIGHT d4_dim_u64_r262k)"
        do_case u64_sparse_right  "${mt}" "$(join_q d4_probe_u64_nomatch RIGHT d4_dim_u64_sparse)"

        # --- strictness: FULL also traverses, INNER does not --------------------------------
        do_case k16_top64_full  "${mt}" "$(join_q d4_probe_k16_nomatch FULL  d4_dim_k16_top64)"
        do_case k16_top64_inner "${mt}" "$(join_q d4_probe_k16_nomatch INNER d4_dim_k16_top64)"

        # --- a real probe side, to put the penalty on a scale --------------------------------
        do_case k16_top64_right_10m "${mt}" "$(join_q d4_probe_k16_10m RIGHT d4_dim_k16_top64)"
        do_case u64_r257_right_1m   "${mt}" "$(join_q d4_probe_u64_1m  RIGHT d4_dim_u64_r257)"

        # --- as shipped: the runtime filter this family otherwise disables is built from
        # --- exactly these maps, so one case has to run with it on.
        do_case k16_top64_right_rtf "${mt}" "$(join_q d4_probe_k16_nomatch RIGHT d4_dim_k16_top64)" \
            --enable_join_runtime_filters=1

        # --- the second traversal: tryRerangeRightTableDataImpl's forEachMapped, on an INNER
        # --- join. Off by default, and only 250 keys x 40 rows satisfies its gates.
        do_case k16_rerange_inner "${mt}" "$(join_q d4_probe_k16_nomatch INNER d4_rerange_k16)" \
            --allow_experimental_join_right_table_sorting=1
        do_case k16_rerange_inner_10m "${mt}" "$(join_q d4_probe_k16_10m INNER d4_rerange_k16)" \
            --allow_experimental_join_right_table_sorting=1
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

# Optional wider counters, only for the two cases where the question is memory behaviour.
if [ "${PERF:-0}" = "1" ]; then
    perf_point d4_top64_mem "${PERF_MEM}" "${SYNTH_DB}" \
        "$(join_q d4_probe_k16_nomatch RIGHT d4_dim_k16_top64)" unified_hash 16
    perf_point d4_r262k_mem "${PERF_MEM}" "${SYNTH_DB}" \
        "$(join_q d4_probe_u64_nomatch RIGHT d4_dim_u64_r262k)" unified_hash 16
fi

collect_stages
collect_qlog
summarize

cat <<'EOF'

How to read this
  * Confirm maptypes.txt first: key16 / key8 / range16_key64 / range17_key64 / range18_key64 as
    listed in SPEC_MAPS.md section 1.4, and `Converted join hash map to fixed hash map` with the
    range and key counts that section predicts. A case showing anything else is void.
  * stages.tsv column nonjoin_us is the measurement. Compare unified_hash against hash for the
    u64_* rows -- parallel_hash never converts key64 to a range* map at all, so its gap there is
    a different algorithm, not D4.
  * The signature that confirms the mechanism rather than merely observing a difference:
    k16_top64 minus k16_bot64 grows with max_threads (leading penalty, once per non-joined
    stream) while k16_bot64 minus its baseline stays flat (trailing penalty, paid once).
  * k16_top64_inner and u64_sparse_right must show no gap. If they do, something other than the
    fixed-map traversal is being measured.
EOF

stop_server
trap - EXIT
echo
echo "M_D4_DONE"
