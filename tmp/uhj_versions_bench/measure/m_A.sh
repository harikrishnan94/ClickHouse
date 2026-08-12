#!/usr/bin/env bash
# A - `Unified::scatterBlockBySlot`: an extra per-row pass over every build block.
#
# Mechanism. When `slots > 1`, `UnifiedHashJoin/HashJoin.cpp:1011` routes every build row through a
# full key getter to compute a routing hash, derives a bucket and a slot, then makes a second pass
# filling the per-slot index columns (and a third materialising `dense_keys` when the summed key
# width is <= 8 bytes and the incoming selector is the identity range). Serial `hash` does no
# scatter at all; `parallel_hash` does a different one (`ConcurrentHashJoin::dispatchBlock`).
#
# Metric. Retired instructions per build row, not wall time: instructions are additive across build
# threads, so they are directly comparable between max_threads=1 (no scatter) and max_threads=16
# (scatter), which wall time is not. Every query shape is build-only - a one-row probe side - so the
# denominator is exactly the number of scattered rows.
#
# Needs no rebuild. See SPEC_HIGH.md for the optional `slotCountForThreads` variant that separates
# the scatter from the slot locking.
#
# Env: ARM=baseline|uhj (default: both), REPEATS, GRID_ROWS, PERF_SECONDS, SKIP_PERF=1,
#      SKIP_CENSUS=1. Idempotent; results accumulate in /mnt/data/uhj_versions_bench/measure/A/.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ID=A
OUT="$(m_out_dir "${ID}")"
REPEATS="${REPEATS:-3}"
GRID_ROWS="${GRID_ROWS:-8000000}"
PERF_ROWS=16000000
PERF_SECONDS="${PERF_SECONDS:-30}"
WANT_ARM="${ARM:-}"
START_TS="$(now_epoch)"

m_take_lock
echo "== m_${ID}: scatterBlockBySlot =="
echo "# out=${OUT} repeats=${REPEATS} grid_rows=${GRID_ROWS}"

## ---------------------------------------------------------------------------------------------
## Query shapes. All build-only: `bench_synth.probe_one` has exactly one row, so essentially all
## the work is read the build table, scatter it, insert it.
##
##   str48   48-byte String key. WORST CASE for A: the key getter materialises a StringRef and the
##           routing hash is a full hash of all 48 bytes, and a variable-width key can never take
##           the `dense_keys` fast path, so the scatter is two full passes with an expensive hash
##           in the first one.
##   keys256 four UInt64 columns packed into a 32-byte key (UInt256HashCRC32). Middle case.
##   u64     one UInt64 key. BEST CASE for A: cheapest key getter, and the summed key width is
##           exactly `sizeof(IColumn::Selector::value_type)`, so `dense_keys` fires and the scatter
##           additionally produces a scattered copy of the key column - which the insert then reads
##           instead of going back through the selector.
## ---------------------------------------------------------------------------------------------
sql_shape() {   # sql_shape <shape> <strictness> <rows>
    local shape="$1" strict="$2" rows="$3" join
    case "${strict}" in
        all_inner) join="INNER JOIN" ;;
        all_left)  join="LEFT JOIN" ;;
        any_left)  join="LEFT ANY JOIN" ;;
        *) echo "bad strictness ${strict}" >&2; return 1 ;;
    esac
    case "${shape}" in
        str48)
            echo "SELECT count() FROM ${SYNTH_DB}.probe_one AS p ${join}
                  (SELECT k FROM ${SYNTH_DB}.build_str48 WHERE id < ${rows}) AS r
                  ON toString(p.id) = r.k" ;;
        keys256)
            echo "SELECT count() FROM ${SYNTH_DB}.probe_one AS p ${join}
                  (SELECT a, b, c, d FROM ${SYNTH_DB}.build_keys256 WHERE id < ${rows}) AS r
                  ON p.id = r.a AND p.id = r.b AND p.id = r.c AND p.id = r.d" ;;
        u64)
            echo "SELECT count() FROM ${SYNTH_DB}.probe_one AS p ${join}
                  (SELECT id AS k FROM ${SYNTH_DB}.build_u64 WHERE id < ${rows}) AS r
                  ON p.id = r.k" ;;
        *) echo "bad shape ${shape}" >&2; return 1 ;;
    esac
}

arm_algos() {
    case "$1" in
        baseline) echo "hash parallel_hash" ;;
        uhj)      echo "unified_hash" ;;
    esac
}

enable_counters() {
    if perfev_available; then
        MEASURE_SETTINGS=("${PERFEV_SETTINGS[@]}")
        echo "# per-query hardware counters: available"
    else
        MEASURE_SETTINGS=()
        echo "# per-query hardware counters: NOT available - the instruction columns will be 0."
        echo "#   Pass 2 (server-wide perf stat) still provides instruction counts for the corners."
    fi
}

## ---------------------------------------------------------------------------------------------
## Pass 1 - the average-case grid: key type x strictness x max_threads, one row each.
## ---------------------------------------------------------------------------------------------
GRID="${OUT}/grid.tsv"

run_grid_for_arm() {
    local arm="$1" algo shape strict mt sql tag best times chosen streams flag
    start_server "${arm}"
    ensure_synth
    enable_counters
    for algo in $(arm_algos "${arm}"); do
        flag="$(algo_flag "${algo}")"
        for shape in str48 keys256 u64; do
            for strict in all_inner any_left; do
                sql="$(sql_shape "${shape}" "${strict}" "${GRID_ROWS}")"
                for mt in 1 2 4 16; do
                    tag="mA_${algo}_${shape}_${strict}_mt${mt}"
                    tsv_prune "${GRID}" "${tag}"
                    q_warm "${SYNTH_DB}" "${sql}" "${flag}" "--max_threads=${mt}"
                    chosen="$(q_algorithm "${SYNTH_DB}" "${sql}" "${flag}" "--max_threads=${mt}")"
                    streams="$(q_build_streams "${SYNTH_DB}" "${sql}" "${flag}" "--max_threads=${mt}")"
                    read -r best times <<<"$(q_best "${REPEATS}" "${tag}" "${SYNTH_DB}" "${sql}" \
                                              "${flag}" "--max_threads=${mt}")"
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "${algo}" "${shape}" "${strict}" "${mt}" "${chosen}" "${streams}" \
                        "${best}" "${times}" "${tag}" >> "${GRID}"
                    echo "  ${algo} ${shape} ${strict} mt=${mt} plan=${chosen} streams=${streams} best=${best}s"
                done
            done
        done
    done
    qlog_agg "mA_" "${START_TS}" "${OUT}/qlog_${arm}.tsv"
    pplog_dump "mA_" "${START_TS}" "${OUT}/pplog_${arm}.tsv"
    stop_server
}

## ---------------------------------------------------------------------------------------------
## Pass 2 - server-wide `perf stat` on the worst-case corner, so the headline number does not
## depend on per-query counters being permitted by perf_event_paranoid.
##
##   mt=1   no scatter on any arm (UHJ slots == 1; ConcurrentHashJoin short-circuits dispatchBlock
##          when num_shards == 1)
##   mt=16  scatter on unified_hash and on parallel_hash; serial `hash` still has none, and its
##          pipeline still has one build stream, so it is the "no parallel machinery" reference
## ---------------------------------------------------------------------------------------------
PERFTSV="${OUT}/perf_corners.tsv"
LOOP_SQL=""
LOOP_ARGS=()
loop_once() { q_warm "${SYNTH_DB}" "${LOOP_SQL}" "${LOOP_ARGS[@]}"; }

run_perf_for_arm() {
    local arm="$1" algo mt prefix iters instr cycles llc dtlb
    start_server "${arm}"
    MEASURE_SETTINGS=()
    LOOP_SQL="$(sql_shape str48 all_inner "${PERF_ROWS}")"
    for algo in $(arm_algos "${arm}"); do
        tsv_prune_field "${PERFTSV}" 1 "${algo}"
        for mt in 1 16; do
            LOOP_ARGS=("$(algo_flag "${algo}")" "--max_threads=${mt}")
            prefix="${OUT}/perf_${algo}_mt${mt}"
            q_warm "${SYNTH_DB}" "${LOOP_SQL}" "${LOOP_ARGS[@]}"
            echo "  perf ${algo} mt=${mt}: core group"
            iters="$(perf_window "${prefix}_core" "${PERF_CORE}" "${PERF_SECONDS}" loop_once | sed 's/iters=//')"
            instr="$(perf_value "${prefix}_core" inst_retired)"
            cycles="$(perf_value "${prefix}_core" cpu_cycles)"
            echo "  perf ${algo} mt=${mt}: mem group"
            perf_window "${prefix}_mem" "${PERF_MEM}" "${PERF_SECONDS}" loop_once >/dev/null
            llc="$(perf_value "${prefix}_mem" ll_cache_miss_rd)"
            dtlb="$(perf_value "${prefix}_mem" dtlb_walk)"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${algo}" "${mt}" "${iters}" "${instr}" "${cycles}" "${llc}" "${dtlb}" "${PERF_ROWS}" >> "${PERFTSV}"
            awk -v a="${algo}" -v m="${mt}" -v i="${iters}" -v n="${instr}" -v c="${cycles}" -v r="${PERF_ROWS}" \
                'BEGIN { if (i > 0) printf "    -> %s mt=%s: %.1f instr/row, %.1f cyc/row over %d iters\n", a, m, n/(i*r), c/(i*r), i }'
        done
    done
    stop_server
}

## ---------------------------------------------------------------------------------------------
## Pass 3 - which real-world queries pay for the scatter, and how much.
##
## Exposure is exactly JoinBuildTableRowCount whenever max_threads > 1, because the scatter runs
## once per build row per clause. The census executes every query in all four suites once on the
## UHJ arm and reports the implied scatter budget at the ~46 instructions/row previously measured
## on JOB q64, as a share of the query's own retired instructions.
## ---------------------------------------------------------------------------------------------
run_census() {
    start_server uhj
    enable_counters
    Q_TIMEOUT="${CENSUS_TIMEOUT:-180}"
    census_exec mAc --join_algorithm=unified_hash
    Q_TIMEOUT=0
    flush_logs
    client --query "
        SELECT splitByChar('_', query_id)[2]                                       AS suite,
               splitByChar('_', query_id)[3]                                       AS q,
               ProfileEvents['JoinBuildTableRowCount']                             AS build_rows,
               ProfileEvents['PerfInstructions']                                   AS instructions,
               toUInt64(46 * build_rows)                                           AS scatter_instr_est,
               if(instructions > 0, round(100. * 46 * build_rows / instructions, 2), 0) AS scatter_pct_est,
               round(query_duration_ms / 1000, 4)                                  AS sec
        FROM system.query_log
        WHERE type = 'QueryFinish' AND query_id LIKE 'mAc\\_%' AND event_time >= toDateTime(${START_TS})
        ORDER BY scatter_pct_est DESC, build_rows DESC
        FORMAT TSVWithNames" > "${OUT}/census.tsv" 2>&1 || true
    stop_server
}

## ---------------------------------------------------------------------------------------------
main() {
    [ -s "${GRID}" ]    || printf 'algo\tshape\tstrictness\tmax_threads\tplan\tbuild_streams\tbest_sec\tall_sec\ttag\n' > "${GRID}"
    [ -s "${PERFTSV}" ] || printf 'algo\tmax_threads\titers\tinst_retired\tcpu_cycles\tll_cache_miss_rd\tdtlb_walk\tbuild_rows\n' > "${PERFTSV}"

    hr; echo "PASS 1: grid (best of ${REPEATS}, ${GRID_ROWS} build rows, one probe row)"; hr
    if want_arm baseline; then run_grid_for_arm baseline; fi
    if want_arm uhj;      then run_grid_for_arm uhj;      fi

    if [ "${SKIP_PERF:-0}" != 1 ]; then
        hr; echo "PASS 2: server-wide perf stat, worst-case corner (str48 / ALL INNER / ${PERF_ROWS} rows)"; hr
        if want_arm baseline; then run_perf_for_arm baseline; fi
        if want_arm uhj;      then run_perf_for_arm uhj;      fi
    fi

    if [ "${SKIP_CENSUS:-0}" != 1 ] && want_arm uhj; then
        hr; echo "PASS 3: real-world census (four suites, UHJ arm)"; hr
        run_census
    fi

    hr; echo "SUMMARY m_${ID} - scatterBlockBySlot"; hr
    echo "-- best-of-${REPEATS} wall time, build-only queries (s) --"
    awk -F'\t' 'NR > 1 { printf "%-14s %-8s %-10s mt=%-3s %-20s %s\n", $1, $2, $3, $4, $5, $7 }' "${GRID}" | sort
    local f
    for f in uhj baseline; do
        [ -s "${OUT}/qlog_${f}.tsv" ] || continue
        echo
        echo "-- ${f}: retired instructions per build row (per-query counters) --"
        awk -F'\t' 'NR > 1 { printf "%-44s %12s\n", $1, $12 }' "${OUT}/qlog_${f}.tsv"
    done
    if [ "$(wc -l < "${PERFTSV}")" -gt 1 ]; then
        echo
        echo "-- server-wide perf, str48 / ALL INNER, per build row --"
        awk -F'\t' 'NR > 1 && $3 > 0 { printf "%-14s mt=%-3s %9.1f instr %9.1f cyc %8.3f llc_miss %8.4f dtlb_walk\n",
                                       $1, $2, $4/($3*$8), $5/($3*$8), $6/($3*$8), $7/($3*$8) }' "${PERFTSV}"
    fi
    if [ -s "${OUT}/census.tsv" ]; then
        echo
        echo "-- real-world: top 20 queries by estimated scatter share (46 instr/build row) --"
        head -21 "${OUT}/census.tsv" | column -t -s $'\t' || head -21 "${OUT}/census.tsv"
    fi
    hr
    echo "artifacts: ${OUT}"
    echo "M_${ID}_DONE"
}

main "$@"
