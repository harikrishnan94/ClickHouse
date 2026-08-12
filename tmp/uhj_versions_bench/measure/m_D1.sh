#!/usr/bin/env bash
# D1 - `parallel_hash_join_threshold` is bypassed for `unified_hash`.
#
# Mechanism. `Planner/PlannerJoins.cpp:1244` guards the `rhs_size_estimation >=
# parallel_hash_join_threshold` decision with `&& !unified`, so `unified_hash` never sees the gate
# and is always constructed with the raw `params.max_threads` (line 1259-1264). Baseline
# `join_algorithm='direct,parallel_hash,hash'` runs the SERIAL `HashJoin` on every join whose right
# side is estimated below 100 000 rows; `unified_hash` runs the parallel one - 256-bucket maps,
# `slotCountForThreads(max_threads)` slots and arenas, the scatter of divergence A, and
# `max_threads` FillingRightJoinSide streams instead of one.
#
# Needs no rebuild: `parallel_hash_join_threshold` is a setting, so the baseline arm can be made to
# emulate either side of the gate, and that is exactly what separates the DECISION from the
# IMPLEMENTATION:
#
#   B-gate  baseline, join_algorithm=direct,parallel_hash,hash, threshold=100000 (default)
#   B-par   baseline, join_algorithm=direct,parallel_hash,hash, threshold=0      (always parallel)
#   B-ser   baseline, join_algorithm=hash                                        (always serial)
#   U       uhj,      join_algorithm=unified_hash                                (always parallel)
#
# (B-par - B-gate) is the price of ignoring the gate, priced in the baseline's own implementation.
# (U - B-gate) is what a user actually experiences when switching to unified_hash.
#
# Env: ARM=baseline|uhj (default: both), REPEATS, SKIP_CENSUS=1.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ID=D1
OUT="$(m_out_dir "${ID}")"
REPEATS="${REPEATS:-5}"
WANT_ARM="${ARM:-}"
START_TS="$(now_epoch)"

m_take_lock
echo "== m_${ID}: parallel_hash_join_threshold bypass =="
echo "# out=${OUT} repeats=${REPEATS}"

## ---------------------------------------------------------------------------------------------
## Configurations. Each is (arm, code, extra settings) - the code is underscore-free so query ids
## stay parseable with splitByChar('_', ...).
## ---------------------------------------------------------------------------------------------
cfg_args() {
    case "$1" in
        bgate) echo "--join_algorithm=direct,parallel_hash,hash --parallel_hash_join_threshold=100000" ;;
        bpar)  echo "--join_algorithm=direct,parallel_hash,hash --parallel_hash_join_threshold=0" ;;
        bser)  echo "--join_algorithm=hash" ;;
        u)     echo "--join_algorithm=unified_hash" ;;
        # UHJ has no way to express "serial for a small build side" other than lowering max_threads
        # for the whole query, which also serialises the probe and the scan. Measured so the cost of
        # that workaround is on the record.
        umt1)  echo "--join_algorithm=unified_hash --max_threads=1" ;;
    esac
}
arm_cfgs() { case "$1" in baseline) echo "bgate bpar bser" ;; uhj) echo "u umt1" ;; esac; }

## ---------------------------------------------------------------------------------------------
## Shape 1 - the build-size sweep across the threshold.
##
## `bench_synth.dim_<N>` are separate physical tables so that `rhs_size_estimation` is the table's
## own row count and the gate decision is unambiguous. `probe_10m`'s keys are 0..999, which every
## dim contains, so each probe row matches exactly one build row at every N and the join output is
## a constant 10 M rows - the only thing that varies across the sweep is the build side.
## ---------------------------------------------------------------------------------------------
sql_sweep() {   # sql_sweep <dim rows>
    echo "SELECT count() FROM ${SYNTH_DB}.probe_10m AS p
          INNER JOIN ${SYNTH_DB}.dim_$1 AS d ON p.k = d.id"
}

## ---------------------------------------------------------------------------------------------
## Shape 2 - the worst case: a join whose entire cost IS the join setup.
##
## 100 build rows and 100 probe rows. Serial `hash` allocates one 256-cell flat table (~4 KiB) and
## one arena, and the pipeline gets one FillingRightJoinSide. `unified_hash` at max_threads=16
## allocates a 256-bucket map (256 x 256 cells, ~1 MiB), 16 arenas, 16 bucket locks and 16 build
## streams, and runs the scatter - to insert 100 rows. Nothing amortises any of it.
##
## Shape 2b multiplies that by eight by joining a fact table against eight tiny dimensions in one
## query, which is the shape a real star-schema dashboard query has.
## ---------------------------------------------------------------------------------------------
sql_tiny() {
    echo "SELECT count() FROM ${SYNTH_DB}.tiny_fact AS f
          INNER JOIN ${SYNTH_DB}.tiny_dim AS d ON f.k = d.id"
}
sql_star8() {
    local i j=""
    for i in 1 2 3 4 5 6 7 8; do
        j="${j} INNER JOIN ${SYNTH_DB}.tiny_dim AS d${i} ON f.k = d${i}.id"
    done
    echo "SELECT count() FROM ${SYNTH_DB}.tiny_fact AS f ${j}"
}
# Nine relations is within query_plan_optimize_join_order_limit, so without this the two arms could
# be handed different join orders and the comparison would measure the optimizer, not the join.
STAR8_ARGS=(--query_plan_optimize_join_order_limit=0)
## Shape 3 - a realistic "big fact, small dimension" join: 10 M probe rows against a 1 000-row
## dimension. This is the single most common join shape in the loaded suites and sits far below the
## threshold, so baseline runs it serially and unified_hash does not.
sql_realistic() {
    echo "SELECT count(), sum(d.pad) FROM ${SYNTH_DB}.probe_10m AS p
          INNER JOIN ${SYNTH_DB}.dim_1000 AS d ON p.k = d.id"
}

enable_counters() {
    if perfev_available; then MEASURE_SETTINGS=("${PERFEV_SETTINGS[@]}"); echo "# per-query hardware counters: available"
    else MEASURE_SETTINGS=(); echo "# per-query hardware counters: NOT available"; fi
}

SWEEP="${OUT}/sweep.tsv"
TINY="${OUT}/tiny.tsv"

run_for_arm() {
    local arm="$1" cfg n sql tag best times chosen streams shape reps
    local -a args extra
    start_server "${arm}"
    ensure_synth
    enable_counters

    for cfg in $(arm_cfgs "${arm}"); do
        # cfg_args returns several flags in one string; splitting them is the point.
        # shellcheck disable=SC2206
        args=( $(cfg_args "${cfg}") )

        for n in 1000 10000 50000 99000 101000 200000 1000000; do
            sql="$(sql_sweep "${n}")"
            tag="mD1s_${cfg}_n${n}"
            tsv_prune "${SWEEP}" "${tag}"
            q_warm "${SYNTH_DB}" "${sql}" "${args[@]}"
            chosen="$(q_algorithm "${SYNTH_DB}" "${sql}" "${args[@]}")"
            streams="$(q_build_streams "${SYNTH_DB}" "${sql}" "${args[@]}")"
            read -r best times <<<"$(q_best "${REPEATS}" "${tag}" "${SYNTH_DB}" "${sql}" "${args[@]}")"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${arm}" "${cfg}" "${n}" "${chosen}" "${streams}" "${best}" "${times}" "${tag}" >> "${SWEEP}"
            echo "  ${cfg} dim=${n} plan=${chosen} streams=${streams} best=${best}s"
        done

        for shape in tiny star8 realistic; do
            extra=()
            case "${shape}" in
                tiny)      sql="$(sql_tiny)" ;;
                star8)     sql="$(sql_star8)"; extra=("${STAR8_ARGS[@]}") ;;
                realistic) sql="$(sql_realistic)" ;;
            esac
            tag="mD1w_${cfg}_${shape}"
            tsv_prune "${TINY}" "${tag}"
            q_warm "${SYNTH_DB}" "${sql}" "${args[@]}" ${extra[@]+"${extra[@]}"}
            chosen="$(q_algorithm "${SYNTH_DB}" "${sql}" "${args[@]}" ${extra[@]+"${extra[@]}"})"
            streams="$(q_build_streams "${SYNTH_DB}" "${sql}" "${args[@]}" ${extra[@]+"${extra[@]}"})"
            # The tiny shapes finish in single-digit milliseconds, so take many more repeats: the
            # quantity of interest is the fixed per-join cost, which is exactly what noise hides.
            reps="${REPEATS}"
            [ "${shape}" = realistic ] || reps=$((REPEATS * 5))
            read -r best times <<<"$(q_best "${reps}" "${tag}" "${SYNTH_DB}" "${sql}" \
                                      "${args[@]}" ${extra[@]+"${extra[@]}"})"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${arm}" "${cfg}" "${shape}" "${chosen}" "${streams}" "${best}" "${times}" "${tag}" >> "${TINY}"
            echo "  ${cfg} ${shape} plan=${chosen} streams=${streams} best=${best}s"
        done
    done

    qlog_agg "mD1" "${START_TS}" "${OUT}/qlog_${arm}.tsv"
    pplog_dump "mD1" "${START_TS}" "${OUT}/pplog_${arm}.tsv"
    stop_server
}

## ---------------------------------------------------------------------------------------------
## Real-world: how many suite queries the gate actually diverts.
##
## The detector is plan-only and therefore free: run `EXPLAIN actions=1` on the baseline arm with
## the default settings and count queries whose plan contains `Algorithm: HashJoin`. Every one of
## those is a join that unified_hash will run in parallel instead. `EXPLAIN PIPELINE` corroborates:
## serial `HashJoin` has one FillingRightJoinSide, the parallel algorithms have max_threads.
## ---------------------------------------------------------------------------------------------
run_census() {
    start_server baseline
    echo "# census: baseline plans with the gate live"
    census_explain "${OUT}/census_baseline_gate.tsv" \
        --join_algorithm=direct,parallel_hash,hash --parallel_hash_join_threshold=100000
    echo "# census: baseline plans with the gate disabled"
    census_explain "${OUT}/census_baseline_nogate.tsv" \
        --join_algorithm=direct,parallel_hash,hash --parallel_hash_join_threshold=0
    stop_server

    start_server uhj
    echo "# census: unified_hash plans"
    census_explain "${OUT}/census_uhj.tsv" --join_algorithm=unified_hash
    stop_server

    # A query "diverges" when the baseline plan keeps a bare `HashJoin` (the serial one). The regex
    # anchors on the comma-joined list so that ConcurrentHashJoin and UnifiedHashJoin do not match.
    awk -F'\t' 'NR > 1 {
            serial = ($3 ~ /(^|,)HashJoin(,|$)/)
            printf "%s\t%s\t%s\t%s\t%s\n", $1, $2, $3, $4, (serial ? "DIVERGES" : "-")
        }' "${OUT}/census_baseline_gate.tsv" > "${OUT}/census_diverging.tsv"
    {
        printf 'suite\ttotal\twith_serial_hashjoin\n'
        awk -F'\t' '{ tot[$1]++; if ($5 == "DIVERGES") div[$1]++ }
                    END { for (s in tot) printf "%s\t%d\t%d\n", s, tot[s], div[s] + 0 }' \
            "${OUT}/census_diverging.tsv" | sort
    } > "${OUT}/census_summary.tsv"
}

## ---------------------------------------------------------------------------------------------
main() {
    [ -s "${SWEEP}" ] || printf 'arm\tcfg\tdim_rows\tplan\tbuild_streams\tbest_sec\tall_sec\ttag\n' > "${SWEEP}"
    [ -s "${TINY}" ]  || printf 'arm\tcfg\tshape\tplan\tbuild_streams\tbest_sec\tall_sec\ttag\n' > "${TINY}"

    hr; echo "PASS 1: build-size sweep across parallel_hash_join_threshold, plus the tiny-join worst cases"; hr
    if want_arm baseline; then run_for_arm baseline; fi
    if want_arm uhj;      then run_for_arm uhj;      fi

    if [ "${SKIP_CENSUS:-0}" != 1 ]; then
        hr; echo "PASS 2: plan census over the four suites (EXPLAIN only, nothing executed)"; hr
        run_census
    fi

    hr; echo "SUMMARY m_${ID} - parallel_hash_join_threshold bypass"; hr
    echo "-- build-size sweep: 10 M probe rows, one match per probe row at every size --"
    printf '%-6s %-10s %-24s %-8s %s\n' cfg dim_rows plan streams best_sec
    awk -F'\t' 'NR > 1 { printf "%-6s %-10s %-24s %-8s %s\n", $2, $3, $4, $5, $6 }' "${SWEEP}" | sort -k2,2n -k1,1
    echo
    echo "-- worst cases: the whole query is join setup --"
    printf '%-6s %-11s %-24s %-8s %s\n' cfg shape plan streams best_sec
    awk -F'\t' 'NR > 1 { printf "%-6s %-11s %-24s %-8s %s\n", $2, $3, $4, $5, $6 }' "${TINY}" | sort -k2,2 -k1,1
    if [ -s "${OUT}/census_summary.tsv" ]; then
        echo
        echo "-- real world: suite queries whose baseline plan keeps a SERIAL HashJoin --"
        column -t -s $'\t' "${OUT}/census_summary.tsv" || cat "${OUT}/census_summary.tsv"
        echo
        echo "   (per-query detail: ${OUT}/census_diverging.tsv;"
        echo "    unified_hash's own plans: ${OUT}/census_uhj.tsv)"
    fi
    hr
    echo "artifacts: ${OUT}"
    echo "M_${ID}_DONE"
}

main "$@"
