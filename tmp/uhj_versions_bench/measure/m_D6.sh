#!/usr/bin/env bash
# D6 - `use_offset` became a template parameter driven by `JoinFeatures::need_flags`, so UHJ skips
#      `offsetInternal` per probe row when the join needs no used flags. Baseline hard-codes
#      `constexpr bool use_offset = true` (`HashJoin/KeyGetter.h:19`).
#
# Mechanism, stated precisely, because it decides the worst case. For every non-LowCardinality key
# getter the offset is computed inside the shared `ColumnsHashingImpl::findKeyImpl`:
#
#     size_t offset = 0;
#     if constexpr (FindResult::has_offset)
#         offset = it ? data.offsetInternal(it) : 0;
#
# `has_offset` comes from the getter's `need_offset` template parameter, which is `use_offset` in
# baseline and `needs_offset = JoinFeatures<KIND, STRICTNESS, Maps>::need_flags` in UHJ
# (`UnifiedHashJoin/HashJoinMethods.h:135`). So:
#   * the saving is per *matched* probe row (a miss computes nothing on either arm);
#   * it is the whole of `offsetInternal`, which for a two-level map is a cell `isZero` test, a
#     re-hash of the stored key (`getBucketFromHash(ptr->getHash(*this))`), a `std::call_once`
#     acquire load and a prefix-sum add - so it scales with how expensive the key's hash is;
#   * for a serial (flat) map `offsetInternal` is `ptr - buf + 1`, so the saving nearly vanishes.
# `need_flags` is false for INNER/LEFT ALL, LEFT ANY, LEFT and INNER SEMI/ANTI and ASOF, and true
# for every RIGHT/FULL kind and for INNER ANY (`UnifiedHashJoin/joinDispatch.h`).
#
# Worst case therefore: a probe-heavy INNER ALL join, 100 % match rate, `two_level_keys256` map
# (four UInt64 key columns -> a 32-byte key hashed with UInt256HashCRC32, the dearest re-hash of any
# join map), max_threads > 1. Contrast cells: the same thing on `key64` (cheap re-hash) and at
# max_threads = 1 (flat map, trivial offset).
#
# Isolation. No setting toggles `use_offset`, and the raw arm-to-arm gap on a flags-off join also
# contains the batched-probe rewrite that this measurement family excludes. So the mechanism is
# isolated by a difference-in-differences over "does this join need flags", which cancels everything
# that is common to both shapes:
#
#     D6_residual = (uhj_flagson - uhj_flagsoff) - (base_flagson - base_flagsoff)
#
# On baseline both shapes pay `offsetInternal`, so its bracket contains only the flag writes; on UHJ
# only the flags-on shape pays it, so the difference of the two brackets is what an offset
# computation costs. Baseline pays that same amount on flags-off joins and throws it away, which is
# exactly what D6 removes. Two independent flags-on comparators are used (INNER ANY and RIGHT SEMI)
# because each has its own second-order difference from INNER ALL; agreement between them is the
# evidence that the DiD is measuring the offset and not the comparator.
#
# All three shapes are run 1:1 (one probe row per distinct build key) so that they emit exactly the
# same number of rows and the DiD is not contaminated by emission volume. Note that INNER ALL with
# unique build keys is promoted to RightAny at `onBuildPhaseFinish` on both arms - also flags-off,
# so the promotion does not affect what is being isolated.
#
# Results: /mnt/data/uhj_versions_bench/measure/D6/
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_probe_common.sh"

ID=D6
OUT="$(m_out_dir "${ID}")"
REPEATS="${REPEATS:-3}"
MT_LIST="${MT_LIST:-1 8 16}"
WORST_MT="${WORST_MT:-8}"

RES="${OUT}/results.tsv"
MAPS="${OUT}/maptypes.tsv"

m_take_lock

suite_census "${OUT}/suite_census.tsv"
suite_census_summary "${OUT}/suite_census.tsv" | tee "${OUT}/suite_census_summary.txt"
echo "# D6 exposure: every join whose kind/strictness needs no used flags, i.e. everything in the"
echo "#   suites except the FULL joins - see the FULL column above."

N1="${PB_K256_BUILD}"      # 1:1 cells: probe rows = build rows = distinct keys
NP="${PB_K256_PROBE}"      # probe-heavy cell: NP/N1 probe rows per build key

## -------------------------------------------------------------------------------------------
## Query shapes. <key> is u64 (key64 / two_level_key64) or k256 (keys256 / two_level_keys256).
##
##   innerall   flags off - the shape D6 makes cheaper
##   anyinner   flags on  - INNER ANY: MapsOne, setUsedOnce, one row per right row
##   rightsemi  flags on  - RIGHT SEMI: MapsAll like innerall, no non-joined phase (Semi is
##                          excluded by JoinCommon::hasNonJoinedBlocks), one row per right row
##   probeheavy flags off - the headline worst case: NP probe rows, all matching
## -------------------------------------------------------------------------------------------
sql_cell() {   # sql_cell <key> <shape>
    local key="$1" shape="$2" on l r
    if [ "${key}" = k256 ]; then
        r="pb_k256_build"; l="pb_k256_probe"
        on="l.a = r.a AND l.b = r.b AND l.c = r.c AND l.d = r.d"
    else
        r="pb_u64_build"; l="pb_u64_probe"
        on="l.a = r.a"
    fi
    case "${shape}" in
    innerall)   echo "SELECT count() FROM (SELECT * FROM ${l} WHERE i < ${N1}) AS l
                      INNER JOIN ${r} AS r ON ${on}" ;;
    anyinner)   echo "SELECT count() FROM (SELECT * FROM ${l} WHERE i < ${N1}) AS l
                      ANY INNER JOIN ${r} AS r ON ${on}" ;;
    rightsemi)  echo "SELECT count() FROM (SELECT * FROM ${l} WHERE i < ${N1}) AS l
                      RIGHT SEMI JOIN ${r} AS r ON ${on}" ;;
    probeheavy) echo "SELECT count() FROM ${l} AS l INNER JOIN ${r} AS r ON ${on}" ;;
    *) echo "unknown shape '${shape}'" >&2; return 1 ;;
    esac
}

run_cell() {   # run_cell <algo> <key> <shape> <max_threads>
    local algo="$1" key="$2" shape="$3" mt="$4"
    local tag="pb${ID}_$(algo_code "${algo}")_${key}${shape}_mt${mt}"
    local sql best all instr probe_rows result ipr
    sql="$(sql_cell "${key}" "${shape}")"
    q_warm "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}"
    read -r best all <<<"$(q_best "${REPEATS}" "${tag}" "${SYNTH_DB}" "${sql}" \
        "--join_algorithm=${algo}" "--max_threads=${mt}")"
    instr="$(qlog_metric "${tag}" "toUInt64(median(ProfileEvents['PerfInstructions']))")"
    probe_rows="$(qlog_metric "${tag}" "toUInt64(median(ProfileEvents['JoinProbeTableRowCount']))")"
    result="$(qlog_metric "${tag}" "toUInt64(median(ProfileEvents['JoinResultRowCount']))")"
    ipr="$(awk -v a="${instr}" -v b="${probe_rows}" 'BEGIN { printf "%.3f", (b > 0) ? a / b : 0 }')"
    tsv_prune "${RES}" "${tag}"
    tsv "${RES}" "${tag}" "${CUR_ARM}" "${algo}" "${key}" "${shape}" "${mt}" "${best}" "${instr}" \
        "${probe_rows}" "${result}" "${ipr}" "${all}"
    printf '  %-40s best=%-8s instr/probe_row=%-9s probe_rows=%-10s result_rows=%s\n' \
        "${tag}" "${best}" "${ipr}" "${probe_rows}" "${result}"
}

## -------------------------------------------------------------------------------------------
## Arms
## -------------------------------------------------------------------------------------------
tsv_head "${RES}" tag arm algo key shape max_threads best_sec instructions probe_rows result_rows \
    instr_per_probe_row all_times
tsv_head "${MAPS}" tag arm algo key max_threads map_type

T_START="$(now_epoch)"

for arm in baseline uhj; do
    want_arm "${arm}" || continue
    start_server "${arm}"
    ensure_probe_synth k256

    MEASURE_SETTINGS=()
    if perfev_available; then
        MEASURE_SETTINGS=("${PERFEV_SETTINGS[@]}")
    else
        echo "# per-query hardware counters unavailable: instructions per probe row will be 0 and the"
        echo "#   difference-in-differences must be read off best_sec, which is much noisier." >&2
    fi

    if [ "${arm}" = baseline ]; then algos=(parallel_hash hash); else algos=(unified_hash); fi

    for algo in "${algos[@]}"; do
        # `hash` is serial by construction, so only the max_threads=1 row of it means anything;
        # it is kept as the reference for "what a flat map costs".
        if [ "${algo}" = hash ]; then mts="1"; else mts="${MT_LIST}"; fi

        for mt in ${mts}; do
            for key in u64 k256; do
                local_map="$(q_maptype "${SYNTH_DB}" "$(sql_cell "${key}" innerall)" \
                    "--join_algorithm=${algo}" "--max_threads=${mt}")"
                tag="pb${ID}map_$(algo_code "${algo}")_${key}_mt${mt}"
                tsv_prune "${MAPS}" "${tag}"
                tsv "${MAPS}" "${tag}" "${arm}" "${algo}" "${key}" "${mt}" "${local_map}"
                echo "# map: arm=${arm} algo=${algo} key=${key} mt=${mt} -> ${local_map}"
            done
        done

        hr; echo "# ${ID} difference-in-differences cells: arm=${arm} algo=${algo}"
        for mt in ${mts}; do
            for key in u64 k256; do
                for shape in innerall anyinner rightsemi; do
                    run_cell "${algo}" "${key}" "${shape}" "${mt}"
                done
            done
        done

        hr; echo "# ${ID} worst case (probe-heavy INNER ALL): arm=${arm} algo=${algo}"
        for mt in ${mts}; do
            for key in u64 k256; do
                run_cell "${algo}" "${key}" probeheavy "${mt}"
            done
        done
    done

    qlog_agg "pb${ID}" "${T_START}" "${OUT}/querylog_${arm}.tsv"
done

## -------------------------------------------------------------------------------------------
## Summary: the DiD per (key, max_threads), for both flags-on comparators.
## -------------------------------------------------------------------------------------------
{
    hr
    echo "D6 summary - instructions per probe row (1:1 cells, ${N1} probe rows = ${N1} build keys)"
    hr
    awk -F'\t' '
        NR == 1 { next }
        {
            # v[algo, key, mt, shape] = instructions per probe row; t[...] = best wall seconds
            v[$3 SUBSEP $4 SUBSEP $6 SUBSEP $5] = $11
            t[$3 SUBSEP $4 SUBSEP $6 SUBSEP $5] = $7
            seen[$4 SUBSEP $6] = 1
            algos[$3] = 1
        }
        function did(base, uhj, key, mt, on,    b0, b1, u0, u1) {
            b0 = v[base SUBSEP key SUBSEP mt SUBSEP "innerall"]; b1 = v[base SUBSEP key SUBSEP mt SUBSEP on]
            u0 = v[uhj  SUBSEP key SUBSEP mt SUBSEP "innerall"]; u1 = v[uhj  SUBSEP key SUBSEP mt SUBSEP on]
            if (b0 == "" || b1 == "" || u0 == "" || u1 == "") return "n/a"
            return sprintf("%.2f", (u1 - u0) - (b1 - b0))
        }
        END {
            printf "%-5s %4s  %-28s %-28s %10s %10s\n", "key", "mt",
                   "instr/row innerall (b/u)", "instr/row flagson (b/u)", "DiD_any", "DiD_semi"
            for (k in seen) {
                split(k, p, SUBSEP); key = p[1]; mt = p[2]
                base = (v["parallel_hash" SUBSEP key SUBSEP mt SUBSEP "innerall"] != "") ? "parallel_hash" : "hash"
                printf "%-5s %4s  %-28s %-28s %10s %10s\n", key, mt,
                       sprintf("%s / %s", v[base SUBSEP key SUBSEP mt SUBSEP "innerall"],
                                          v["unified_hash" SUBSEP key SUBSEP mt SUBSEP "innerall"]),
                       sprintf("%s / %s", v[base SUBSEP key SUBSEP mt SUBSEP "anyinner"],
                                          v["unified_hash" SUBSEP key SUBSEP mt SUBSEP "anyinner"]),
                       did(base, "unified_hash", key, mt, "anyinner"),
                       did(base, "unified_hash", key, mt, "rightsemi")
            }
            print ""
            print "DiD_* is (uhj_flagson - uhj_flagsoff) - (base_flagson - base_flagsoff) in retired"
            print "instructions per probe row: what one offset computation costs, and therefore what"
            print "D6 saves on every flags-off join. It should be larger for k256 than for u64 (the"
            print "re-hash is the dominant term) and near zero at max_threads=1 (flat map)."
            print "The raw innerall gap (base/uhj) also contains the excluded batched-probe rewrite,"
            print "so it bounds D6 from above and must not be quoted as D6 on its own."
        }' "${RES}"
    echo
    echo "Worst case, probe-heavy INNER ALL (${NP} probe rows, all matching):"
    awk -F'\t' 'NR == 1 || $5 == "probeheavy"' "${RES}" | column -t -s $'\t' 2>/dev/null \
        || awk -F'\t' 'NR == 1 || $5 == "probeheavy"' "${RES}"
    echo
    echo "Map layouts actually measured:"
    column -t -s $'\t' "${MAPS}" 2>/dev/null || cat "${MAPS}"
    echo
    echo "Result-row equality check (all three 1:1 shapes must agree, or the DiD is invalid):"
    awk -F'\t' 'NR > 1 && $5 != "probeheavy" { print $2, $3, $4, $5, $6, "result_rows=" $10 }' "${RES}" | sort
    echo
    echo "Real-world exposure:"
    cat "${OUT}/suite_census_summary.txt"
} | tee "${OUT}/summary.txt"

done_sentinel "${ID}" "${OUT}"
