#!/usr/bin/env bash
# D5 - the non-joined-rows short-circuit (`hasNonJoinedRows` / `updateNonJoinedRowsStatus` /
#      `allOffsetFlagsSet`) is absent from UHJ.
#
# Mechanism. Baseline `HashJoin` computes, once at `onBuildPhaseFinish`
# (`HashJoin/HashJoin.cpp:2388`), whether any right row can be non-joined and caches the answer in
# `has_non_joined_rows`; `ConcurrentHashJoin::getNonJoinedBlocks` (`ConcurrentHashJoin.cpp:559`)
# consults it and, when it is false, does not build the non-joined stream at all. UHJ has none of
# this and always builds `NotJoinedHash`.
#
# What this script is really for. Reading the code first (see SPEC_PROBE.md) shows the short-circuit
# cannot fire for the case it looks like it was written for:
#   * `updateNonJoinedRowsStatus` runs at build-phase finish, before any probe row has set a flag,
#     so `allOffsetFlagsSet()` is evaluated on an all-false array and returns false, making
#     `found_non_joined` true for every non-empty right side;
#   * it is consulted only from the single-level-map branch of
#     `ConcurrentHashJoin::getNonJoinedBlocks`, which under `parallel_hash` is reachable only for
#     `key8` / `key16` maps - a full scan of at most 65537 cells.
# So the measurement has three jobs, in order of importance:
#   1. falsify or confirm that reading, by measuring the non-joined phase in both arms for the
#      worst case (every right row matched): if baseline skipped the scan its
#      NonJoinedBlocksTransform would cost ~0 and UHJ's would not;
#   2. quantify the scan itself - the size of the optimisation neither arm currently gets;
#   3. check the one window where the baseline short-circuit does fire (`rows_to_join == 0`).
#
# Measured arms: `parallel_hash` on clickhouse-baseline against `unified_hash` on clickhouse-uhj,
# both at max_threads > 1. That pair is the only one where the non-joined phase is separately
# attributable: with more than one stream both produce `NonJoinedBlocksTransform` sources, whose
# elapsed_us is exactly the scan. Serial `hash` emits non-joined rows from inside JoiningTransform,
# so for that arm only whole-query time is meaningful, and it is recorded as context, not as the
# comparison.
#
# Results: /mnt/data/uhj_versions_bench/measure/D5/
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_probe_common.sh"

ID=D5
OUT="$(m_out_dir "${ID}")"
REPEATS="${REPEATS:-3}"
SWEEP_REPEATS="${SWEEP_REPEATS:-2}"
WORST_MT="${WORST_MT:-8}"
SWEEP_MT="${SWEEP_MT:-2 8 16}"

RES="${OUT}/results.tsv"
PPRES="${OUT}/processors.tsv"
QLRES="${OUT}/querylog.tsv"
BEHAVE="${OUT}/behaviour.tsv"

m_take_lock

## -------------------------------------------------------------------------------------------
## Phase 0: real-world exposure. No server needed; done first so it is recorded even if the
## measurement is interrupted.
## -------------------------------------------------------------------------------------------
suite_census "${OUT}/suite_census.tsv"
suite_census_summary "${OUT}/suite_census.tsv" | tee "${OUT}/suite_census_summary.txt"
echo "# D5 exposure: only RIGHT and FULL joins can reach it - see the RIGHT/FULL columns above."

## -------------------------------------------------------------------------------------------
## Query shapes.
##
##   anti100 - RIGHT ANTI over a 32 M-cell map where the probe covers every key. Output is zero
##             rows, so the whole non-joined phase is scan-and-discard: the purest possible
##             measurement of "the scan that a working short-circuit would have skipped", and the
##             exact worst case D5 asks for (a very large right side where every right row matched).
##   right100 - the same data as an ordinary RIGHT JOIN, to show the scan cost in proportion to a
##             realistic query rather than in isolation.
##   anti50  - half the keys probed: half the right rows are non-joined, so the scan also emits.
##             This is the control that proves the scan is running in both arms at all.
##   anti0   - a probe that matches nothing: every right row is non-joined. The upper bound on the
##             non-joined phase, and the case where the removed short-circuit would correctly say
##             "yes, there are non-joined rows" in both arms.
##   full100 - FULL JOIN, all matched: the only outer kind that actually occurs in the suites.
## -------------------------------------------------------------------------------------------
HALF=$(( PB_DIM_ROWS / 2 ))

sql_shape() {   # sql_shape <shape>
    case "$1" in
    anti100)  echo "SELECT count() FROM pb_probe_u64 AS l RIGHT ANTI JOIN pb_dim_u64 AS r ON l.k = r.k" ;;
    right100) echo "SELECT count() FROM pb_probe_u64 AS l RIGHT JOIN pb_dim_u64 AS r ON l.k = r.k" ;;
    anti50)   echo "SELECT count() FROM (SELECT k FROM pb_probe_u64 WHERE k < ${HALF}) AS l
                    RIGHT ANTI JOIN pb_dim_u64 AS r ON l.k = r.k" ;;
    anti0)    echo "SELECT count() FROM (SELECT k + 1000000000 AS k FROM pb_probe_u64) AS l
                    RIGHT ANTI JOIN pb_dim_u64 AS r ON l.k = r.k" ;;
    full100)  echo "SELECT count() FROM pb_probe_u64 AS l FULL JOIN pb_dim_u64 AS r ON l.k = r.k" ;;
    *) echo "unknown shape '$1'" >&2; return 1 ;;
    esac
}

# One cell: run it REPEATS times, then pull its aggregates out of the two system logs.
# tag layout: pb<ID>_<algocode>_<shape>_mt<N>   (no '_' inside any component)
run_cell() {   # run_cell <algo> <shape> <max_threads> <repeats>
    local algo="$1" shape="$2" mt="$3" reps="$4"
    local tag="pb${ID}_$(algo_code "${algo}")_${shape}_mt${mt}"
    local sql t0 best all
    sql="$(sql_shape "${shape}")"
    t0="$(now_epoch)"
    q_warm "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}"
    read -r best all <<<"$(q_best "${reps}" "${tag}" "${SYNTH_DB}" "${sql}" \
        "--join_algorithm=${algo}" "--max_threads=${mt}")"

    local nj_us nj_streams nj_zero nj_rows probe_us build_us right_rows result
    nj_us="$(pp_metric "${tag}" NonJoinedBlocksTransform 'toUInt64(sum(elapsed_us) / greatest(count(distinct query_id), 1))')"
    nj_streams="$(pp_metric "${tag}" NonJoinedBlocksTransform 'toUInt64(count() / greatest(count(distinct query_id), 1))')"
    nj_zero="$(pp_metric "${tag}" NonJoinedBlocksTransform 'toUInt64(countIf(output_rows = 0) / greatest(count(distinct query_id), 1))')"
    nj_rows="$(pp_metric "${tag}" NonJoinedBlocksTransform 'toUInt64(sum(output_rows) / greatest(count(distinct query_id), 1))')"
    probe_us="$(pp_metric "${tag}" JoiningTransform 'toUInt64(sum(elapsed_us) / greatest(count(distinct query_id), 1))')"
    build_us="$(pp_metric "${tag}" FillingRightJoinSide 'toUInt64(sum(elapsed_us) / greatest(count(distinct query_id), 1))')"
    right_rows="$(qlog_metric "${tag}" "toUInt64(median(ProfileEvents['JoinBuildTableRowCount']))")"
    result="$(q_scalar "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}")"

    tsv_prune "${RES}" "${tag}"
    tsv "${RES}" "${tag}" "${CUR_ARM}" "${algo}" "${shape}" "${mt}" "${best}" "${nj_us}" "${nj_streams}" \
        "${nj_zero}" "${nj_rows}" "${probe_us}" "${build_us}" "${right_rows}" "${result}" "${t0}" "${all}"
    printf '  %-34s best=%-8s nonjoined_us=%-9s streams=%-3s emitted=%-10s result=%s\n' \
        "${tag}" "${best}" "${nj_us}" "${nj_streams}" "${nj_rows}" "${result}"
}

## -------------------------------------------------------------------------------------------
## The crux behaviour probe: `key8`, every right row matched, `parallel_hash`.
##
## This is the only configuration in which baseline even consults `hasNonJoinedRows()`
## (`ConcurrentHashJoin.cpp:559` is on the single-level-map branch, and `key8`/`key16` are the only
## single-level maps a parallel build can produce). If the short-circuit worked as its name
## suggests, baseline would report zero non-joined stream work here and UHJ would not.
## -------------------------------------------------------------------------------------------
probe_key8() {   # probe_key8 <algo>
    local algo="$1" mt="${WORST_MT}"
    local tag="pb${ID}k8_$(algo_code "${algo}")"
    local sql="SELECT count() FROM (SELECT toUInt8(number % 256) AS k FROM numbers_mt(10000000)) AS l
               RIGHT ANTI JOIN pb_dim_u8 AS r ON l.k = r.k"
    q_warm "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}"
    q_time "${tag}_r1" "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}" >/dev/null
    local maptype nj_us nj_streams
    maptype="$(q_maptype "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}")"
    nj_us="$(pp_metric "${tag}" NonJoinedBlocksTransform 'toUInt64(sum(elapsed_us))')"
    nj_streams="$(pp_metric "${tag}" NonJoinedBlocksTransform 'toUInt64(count())')"
    tsv_prune "${BEHAVE}" "${tag}"
    tsv "${BEHAVE}" "${tag}" "${CUR_ARM}" "${algo}" "key8-all-matched" "${maptype}" "${nj_streams}" "${nj_us}" \
        "single-level map branch: the only place hasNonJoinedRows() is consulted"
    printf '  %-24s map=%-20s nonjoined_streams=%-3s nonjoined_us=%s\n' "${tag}" "${maptype}" "${nj_streams}" "${nj_us}"
}

## -------------------------------------------------------------------------------------------
## The one window where baseline's short-circuit does fire: an empty right side
## (`data->rows_to_join == 0`). FULL JOIN rather than RIGHT, because for RIGHT
## `alwaysReturnsEmptySet()` already stops the probe on both arms and nothing else happens.
## -------------------------------------------------------------------------------------------
probe_empty_right() {   # probe_empty_right <algo>
    local algo="$1" mt="${WORST_MT}"
    local tag="pb${ID}er_$(algo_code "${algo}")"
    local sql="SELECT count() FROM pb_small_l AS l FULL JOIN pb_dim_empty AS r ON l.k = r.k"
    local t res
    t="$(q_time "${tag}_r1" "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}")"
    res="$(q_scalar "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}")"
    tsv_prune "${BEHAVE}" "${tag}"
    tsv "${BEHAVE}" "${tag}" "${CUR_ARM}" "${algo}" "empty-right-side" "-" "-" "$(pp_metric "${tag}" NonJoinedBlocksTransform 'toUInt64(sum(elapsed_us))')" \
        "rows_to_join=0, the only case where hasNonJoinedRows() returns false; result=${res} time=${t}"
    printf '  %-24s time=%-8s result=%s\n' "${tag}" "${t}" "${res}"
}

## -------------------------------------------------------------------------------------------
## Arms
## -------------------------------------------------------------------------------------------
tsv_head "${RES}" tag arm algo shape max_threads best_sec nonjoined_us nonjoined_streams \
    nonjoined_zero_streams nonjoined_rows probe_us build_us right_rows result epoch all_times
tsv_head "${BEHAVE}" tag arm algo probe map_type nonjoined_streams nonjoined_us note

T_START="$(now_epoch)"

for arm in baseline uhj; do
    want_arm "${arm}" || continue
    start_server "${arm}"
    ensure_probe_synth right tiny

    MEASURE_SETTINGS=()
    if perfev_available; then
        MEASURE_SETTINGS=("${PERFEV_SETTINGS[@]}")
        echo "# per-query hardware counters: on"
    else
        echo "# per-query hardware counters: unavailable, wall time and processor times only"
    fi

    if [ "${arm}" = baseline ]; then algos=(parallel_hash hash); else algos=(unified_hash); fi

    for algo in "${algos[@]}"; do
        hr; echo "# ${ID} worst case: arm=${arm} algo=${algo} max_threads=${WORST_MT}"
        for shape in anti100 right100 anti50 anti0 full100; do
            run_cell "${algo}" "${shape}" "${WORST_MT}" "${REPEATS}"
        done

        hr; echo "# ${ID} behaviour probes: arm=${arm} algo=${algo}"
        probe_key8 "${algo}"
        probe_empty_right "${algo}"

        # Average case: match rate x thread count, on the cleanest shape (RIGHT ANTI, no emission
        # at 100%). `hash` is swept too, even though its tail is not separately attributable,
        # because whole-query time is still a valid arm-level comparison.
        hr; echo "# ${ID} sweep: arm=${arm} algo=${algo}"
        for mt in ${SWEEP_MT}; do
            for shape in anti100 anti50 anti0; do
                [ "${mt}" = "${WORST_MT}" ] && continue   # already measured above at more repeats
                run_cell "${algo}" "${shape}" "${mt}" "${SWEEP_REPEATS}"
            done
        done
    done

    qlog_agg  "pb${ID}" "${T_START}" "${OUT}/querylog_${arm}.tsv"
    pplog_agg "pb${ID}" "${T_START}" "${OUT}/processors_${arm}.tsv"
done

cat "${OUT}"/querylog_*.tsv > "${QLRES}" 2>/dev/null || true
cat "${OUT}"/processors_*.tsv > "${PPRES}" 2>/dev/null || true

## -------------------------------------------------------------------------------------------
## Summary
## -------------------------------------------------------------------------------------------
{
    hr
    echo "D5 summary - non-joined scan cost, ${WORST_MT} threads, ${PB_DIM_ROWS} right rows"
    hr
    awk -F'\t' -v mt="${WORST_MT}" '
        NR == 1 { next }
        $5 == mt {
            key = $4
            if ($3 == "parallel_hash") { b[key] = $7; bt[key] = $6; br[key] = $13 }
            if ($3 == "unified_hash")  { u[key] = $7; ut[key] = $6; ur[key] = $13 }
            if ($3 == "hash")          { h[key] = $7; ht[key] = $6 }
        }
        END {
            printf "%-9s %14s %14s %9s %14s %14s %9s %12s\n",
                   "shape", "ph_nonjoin_us", "uh_nonjoin_us", "ratio", "ph_total_s", "uh_total_s", "ratio", "h_total_s"
            n = split("anti100 right100 anti50 anti0 full100", order, " ")
            for (i = 1; i <= n; i++) {
                k = order[i]
                if (!(k in b) && !(k in u)) continue
                r1 = (b[k] > 0) ? u[k] / b[k] : 0
                r2 = (bt[k] > 0) ? ut[k] / bt[k] : 0
                printf "%-9s %14s %14s %9.3f %14s %14s %9.3f %12s\n", k, b[k], u[k], r1, bt[k], ut[k], r2, ht[k]
            }
            print ""
            print "Reading of the numbers:"
            print "  * anti100 is the D5 worst case (every right row matched, zero rows emitted)."
            print "    ph_nonjoin_us near zero would mean the baseline short-circuit fires; a value"
            print "    close to uh_nonjoin_us means both arms scan and D5 costs nothing."
            print "  * anti100 nonjoined_us is also the size of the optimisation that neither arm"
            print "    currently gets, i.e. what a short-circuit evaluated after the probe would save."
            print "  * anti0 bounds the non-joined phase from above (every right row emitted)."
        }' "${RES}"
    echo
    echo "Behaviour probes (${BEHAVE}):"
    column -t -s $'\t' "${BEHAVE}" 2>/dev/null || cat "${BEHAVE}"
    echo
    echo "Real-world exposure:"
    cat "${OUT}/suite_census_summary.txt"
} | tee "${OUT}/summary.txt"

done_sentinel "${ID}" "${OUT}"
