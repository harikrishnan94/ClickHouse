#!/usr/bin/env bash
# D15 - multi-clause (OR) probe prefetch. Baseline builds one prefetcher and points it at `mapv[0]`
#       only ("Only prefetch the first map", `HashJoin/HashJoinMethodsImpl.h:692-705` and `929-943`);
#       UHJ builds one `ProbePrefetch` per clause, each calibrated against its own map
#       (`UnifiedHashJoin/HashJoinMethodsImpl.h:1151-1162` and `1495-1505`).
#
# Both arms gate prefetching on the same runtime test, `shouldUseJoinPrefetch`, which requires the
# user setting `enable_software_prefetch_in_join` and a map whose buffer exceeds L2. That setting is
# what makes this divergence cleanly isolable, and the whole design hangs off it:
#
#     benefit(arm, shape) = T(prefetch off) - T(prefetch on)
#     D15 = benefit(uhj, two clauses) - benefit(baseline, two clauses)
#
# Everything that differs between the arms for reasons other than prefetching - the batched probe
# rewrite, the map layout, D1's planner asymmetry - is present in both terms of each `benefit` and
# cancels. The single-clause shape is the control: there both arms prefetch the one map they have, so
# the two benefits must come out equal. If they do not, the DiD is not measuring prefetching and the
# two-clause number cannot be trusted either.
#
# Worst case: a two-clause disjunction over a map far larger than L2, probe-heavy, so that every
# probe row does two dependent random loads into a map that is not cache resident. `pb_or_r` has
# PB_OR_BUILD cells (>= 8 M, i.e. >= 128 MB of buffer against an L2 of a few MB) and `pb_or_l` has
# PB_OR_PROBE rows. Two variants of the second clause:
#   c2m - the second clause matches the same right row as the first, so both maps return a hit and
#         `KnownRowsHolder` dedupes the output to one row per probe row;
#   c2n - the second clause never matches, so its lookups are pure misses. Misses still take the
#         cache miss, so this is the shape where hiding latency is worth the most and emission
#         volume is smallest.
#
# Confound to keep in view when reading the cross-arm columns (not the DiD): a multi-clause join is
# refused by `TableJoin::allowParallelHashJoin` (`oneDisjunct()` is false), so `join_algorithm =
# 'parallel_hash'` silently runs the serial `HashJoin` here, while `unified_hash` ignores that gate
# and goes 256-bucket parallel (divergence D1). Each cell records the algorithm the planner actually
# chose and the map it built, so the report can say which arms were comparable.
#
# Results: /mnt/data/uhj_versions_bench/measure/D15/
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_probe_common.sh"

ID=D15
OUT="$(m_out_dir "${ID}")"
REPEATS="${REPEATS:-3}"
MT_LIST="${MT_LIST:-1 8}"
PERF_SECS="${PERF_SECS:-20}"

RES="${OUT}/results.tsv"
PLANS="${OUT}/plans.tsv"

m_take_lock

suite_census "${OUT}/suite_census.tsv"
suite_census_summary "${OUT}/suite_census.tsv" | tee "${OUT}/suite_census_summary.txt"
echo "# D15 exposure: the ON-OR column counts suite queries whose ON clause disjoins two equalities,"
echo "#   which is the only thing that produces more than one join clause. Every OR that appears in"
echo "#   these suites is inside a filter predicate, so the expected count is zero."

## -------------------------------------------------------------------------------------------
## Shapes
## -------------------------------------------------------------------------------------------
sql_shape() {   # sql_shape <shape>
    case "$1" in
    c1)  echo "SELECT count() FROM pb_or_l AS l INNER JOIN pb_or_r AS r ON l.x = r.x" ;;
    c2m) echo "SELECT count() FROM pb_or_l AS l INNER JOIN pb_or_r AS r ON l.x = r.x OR l.ym = r.y" ;;
    c2n) echo "SELECT count() FROM pb_or_l AS l INNER JOIN pb_or_r AS r ON l.x = r.x OR l.yn = r.y" ;;
    c3n) echo "SELECT count() FROM pb_or_l AS l INNER JOIN pb_or_r AS r
                ON l.x = r.x OR l.yn = r.y OR l.ym = r.pad" ;;
    *) echo "unknown shape '$1'" >&2; return 1 ;;
    esac
}

run_cell() {   # run_cell <algo> <shape> <max_threads> <prefetch 0|1>
    local algo="$1" shape="$2" mt="$3" pf="$4"
    local tag="pb${ID}_$(algo_code "${algo}")_${shape}_mt${mt}_pf${pf}"
    local sql best all instr probe_rows result cmiss stall ipr cpr
    sql="$(sql_shape "${shape}")"
    local -a extra=("--join_algorithm=${algo}" "--max_threads=${mt}"
                    "--enable_software_prefetch_in_join=${pf}")
    q_warm "${SYNTH_DB}" "${sql}" "${extra[@]}"
    read -r best all <<<"$(q_best "${REPEATS}" "${tag}" "${SYNTH_DB}" "${sql}" "${extra[@]}")"
    instr="$(qlog_metric "${tag}" "toUInt64(median(ProfileEvents['PerfInstructions']))")"
    cmiss="$(qlog_metric "${tag}" "toUInt64(median(ProfileEvents['PerfCacheMisses']))")"
    stall="$(qlog_metric "${tag}" "toUInt64(median(ProfileEvents['PerfStalledCyclesBackend']))")"
    probe_rows="$(qlog_metric "${tag}" "toUInt64(median(ProfileEvents['JoinProbeTableRowCount']))")"
    result="$(qlog_metric "${tag}" "toUInt64(median(ProfileEvents['JoinResultRowCount']))")"
    ipr="$(awk -v a="${instr}" -v b="${probe_rows}" 'BEGIN { printf "%.3f", (b > 0) ? a / b : 0 }')"
    cpr="$(awk -v a="${cmiss}" -v b="${probe_rows}" 'BEGIN { printf "%.4f", (b > 0) ? a / b : 0 }')"
    tsv_prune "${RES}" "${tag}"
    tsv "${RES}" "${tag}" "${CUR_ARM}" "${algo}" "${shape}" "${mt}" "${pf}" "${best}" "${ipr}" \
        "${cpr}" "${stall}" "${probe_rows}" "${result}" "${all}"
    printf '  %-42s best=%-8s instr/row=%-9s cachemiss/row=%-9s result_rows=%s\n' \
        "${tag}" "${best}" "${ipr}" "${cpr}" "${result}"
}

record_plan() {   # record_plan <algo> <shape> <max_threads>
    local algo="$1" shape="$2" mt="$3" sql algos mtype err
    sql="$(sql_shape "${shape}")"
    algos="$(q_algorithm "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}")"
    mtype="$(q_maptype "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}")"
    err="$(q_err "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}")"
    local tag="pb${ID}plan_$(algo_code "${algo}")_${shape}_mt${mt}"
    tsv_prune "${PLANS}" "${tag}"
    tsv "${PLANS}" "${tag}" "${CUR_ARM}" "${algo}" "${shape}" "${mt}" "${algos}" "${mtype}" "${err}"
    printf '  %-34s planner=%-22s map=%-24s %s\n' "${tag}" "${algos}" "${mtype}" "${err}"
}

## -------------------------------------------------------------------------------------------
## Arms
## -------------------------------------------------------------------------------------------
tsv_head "${RES}" tag arm algo shape max_threads prefetch best_sec instr_per_probe_row \
    cachemiss_per_probe_row stall_backend probe_rows result_rows all_times
tsv_head "${PLANS}" tag arm algo shape max_threads planner_algorithms map_type status

T_START="$(now_epoch)"

for arm in baseline uhj; do
    want_arm "${arm}" || continue
    start_server "${arm}"
    ensure_probe_synth orjoin

    MEASURE_SETTINGS=()
    perfev_available && MEASURE_SETTINGS=("${PERFEV_SETTINGS[@]}")

    if [ "${arm}" = baseline ]; then algos=(hash parallel_hash); else algos=(unified_hash); fi

    for algo in "${algos[@]}"; do
        hr; echo "# ${ID} plans: arm=${arm} algo=${algo}"
        for mt in ${MT_LIST}; do
            for shape in c1 c2m c2n c3n; do record_plan "${algo}" "${shape}" "${mt}"; done
        done

        hr; echo "# ${ID} prefetch on/off: arm=${arm} algo=${algo}"
        for mt in ${MT_LIST}; do
            for shape in c1 c2m c2n c3n; do
                for pf in 1 0; do run_cell "${algo}" "${shape}" "${mt}" "${pf}"; done
            done
        done
    done

    # Optional deeper look at the memory hierarchy for the headline shape. The per-query counters
    # above are the primary source; this only adds ll_cache_miss_rd and dtlb_walk, which are not in
    # the per-query list. Skipped silently when sudo perf is not usable.
    if perf_usable; then
        for algo in "${algos[@]}"; do
            for pf in 1 0; do
                PF_ALGO="${algo}" PF_SET="${pf}" perf_loop() {
                    q_warm "${SYNTH_DB}" "$(sql_shape c2n)" "--join_algorithm=${PF_ALGO}" \
                        "--max_threads=8" "--enable_software_prefetch_in_join=${PF_SET}"
                }
                prefix="${OUT}/perf_$(algo_code "${algo}")_c2n_pf${pf}"
                echo "# perf window: algo=${algo} prefetch=${pf}"
                perf_window "${prefix}" "${PERF_MEM}" "${PERF_SECS}" perf_loop
                tsv_prune "${OUT}/perf.tsv" "pb${ID}perf_$(algo_code "${algo}")_pf${pf}"
                tsv_head "${OUT}/perf.tsv" tag arm algo shape prefetch iters cycles instructions \
                    ll_cache_miss_rd dtlb_walk mem_access br_mis_pred
                tsv "${OUT}/perf.tsv" "pb${ID}perf_$(algo_code "${algo}")_pf${pf}" "${arm}" "${algo}" c2n "${pf}" \
                    "$(cat "${prefix}.iters" 2>/dev/null || echo 0)" \
                    "$(perf_value "${prefix}" cpu_cycles)" "$(perf_value "${prefix}" inst_retired)" \
                    "$(perf_value "${prefix}" ll_cache_miss_rd)" "$(perf_value "${prefix}" dtlb_walk)" \
                    "$(perf_value "${prefix}" mem_access)" "$(perf_value "${prefix}" br_mis_pred_retired)"
            done
        done
    else
        echo "# sudo perf not usable: skipping the ll_cache_miss_rd window (per-query counters remain)"
    fi

    qlog_agg "pb${ID}" "${T_START}" "${OUT}/querylog_${arm}.tsv"
done

## -------------------------------------------------------------------------------------------
## Summary
## -------------------------------------------------------------------------------------------
{
    hr
    echo "D15 summary - prefetch benefit per arm, and the difference between arms"
    hr
    awk -F'\t' '
        NR == 1 { next }
        {
            t[$3 SUBSEP $4 SUBSEP $5 SUBSEP $6] = $7          # best_sec  by algo, shape, mt, prefetch
            c[$3 SUBSEP $4 SUBSEP $5 SUBSEP $6] = $9          # cache misses per probe row
            seen[$4 SUBSEP $5] = 1
            algo[$3] = 1
        }
        function ben(a, sh, mt,   on, off) {
            on = t[a SUBSEP sh SUBSEP mt SUBSEP "1"]; off = t[a SUBSEP sh SUBSEP mt SUBSEP "0"]
            if (on == "" || off == "" || on + 0 == 0) return ""
            return (off - on) / on * 100
        }
        END {
            printf "%-5s %4s %22s %22s %22s %14s\n", "shape", "mt",
                   "benefit%% hash", "benefit%% parallel_hash", "benefit%% unified_hash", "DiD(uh-base)"
            n = split("c1 c2m c2n c3n", order, " ")
            for (i = 1; i <= n; i++) {
                sh = order[i]
                for (mt in mts) delete mts[mt]
                for (k in seen) { split(k, p, SUBSEP); if (p[1] == sh) mts[p[2]] = 1 }
                for (mt in mts) {
                    bh = ben("hash", sh, mt); bp = ben("parallel_hash", sh, mt); bu = ben("unified_hash", sh, mt)
                    base = (bp != "") ? bp : bh
                    did = (bu != "" && base != "") ? sprintf("%+.2f", bu - base) : "n/a"
                    printf "%-5s %4s %22s %22s %22s %14s\n", sh, mt,
                           (bh == "" ? "-" : sprintf("%.2f", bh)),
                           (bp == "" ? "-" : sprintf("%.2f", bp)),
                           (bu == "" ? "-" : sprintf("%.2f", bu)), did
                }
            }
            print ""
            print "benefit%% is (T_prefetch_off - T_prefetch_on) / T_prefetch_on, per arm."
            print "c1 is the control: with one clause both arms prefetch the same single map, so the"
            print "DiD there must be ~0. On c2*/c3n a positive DiD means UHJ per-clause prefetching"
            print "wins, a negative one means the extra prefetch instructions cost more than they hide."
            print "Absolute arm-to-arm times on the multi-clause shapes are NOT comparable: see the"
            print "plans table, where parallel_hash falls back to serial hash for OR joins."
        }' "${RES}"
    echo
    echo "Planner choice and map per cell:"
    column -t -s $'\t' "${PLANS}" 2>/dev/null || cat "${PLANS}"
    echo
    echo "Full table:"
    column -t -s $'\t' "${RES}" 2>/dev/null || cat "${RES}"
    [ -s "${OUT}/perf.tsv" ] && { echo; echo "perf windows:"; column -t -s $'\t' "${OUT}/perf.tsv"; }
    echo
    echo "Real-world exposure:"
    cat "${OUT}/suite_census_summary.txt"
} | tee "${OUT}/summary.txt"

done_sentinel "${ID}" "${OUT}"
