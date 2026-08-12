#!/usr/bin/env bash
# B - `BITS_FOR_BUCKET_SERIAL = 0`: the serial UHJ map is one flat table.
#
# Mechanism. `UnifiedHashJoin/HashJoin.h:48` sets `BITS_FOR_BUCKET_SERIAL = 0`, so at
# `max_threads == 1` the `JoinHashMap` alias folds to a single inline `HashMapTable` with
# `HashTableGrowerWithPrecalculation`, and `useTwoLevelMaps` (line 53) makes that a hard function of
# `max_threads` with no setting behind it. `parallel_hash` at `max_threads == 1` instead builds 256
# sub-tables under `TwoLevelHashTableGrower`, so a resize rehashes one bucket rather than the whole
# buffer, and each bucket's probe sequence stays inside a much smaller region.
#
# Rebuild. A clean isolation NEEDS one, because nothing at run time can make serial UHJ build a
# 256-bucket map. Two one-line variants, both in `src/Interpreters/UnifiedHashJoin/HashJoin.h`:
#
#   V-B1 (bucket count only, keeps the flat grower)      line 48
#     -  constexpr Int32 BITS_FOR_BUCKET_SERIAL = 0;
#     +  constexpr Int32 BITS_FOR_BUCKET_SERIAL = 8;
#
#   V-B2 (exactly parallel_hash's layout: 256 buckets + TwoLevelHashTableGrower)   lines 53-56
#     -  inline bool useTwoLevelMaps(size_t max_threads)
#     -  {
#     -      return max_threads > 1;
#     -  }
#     +  inline bool useTwoLevelMaps(size_t /*max_threads*/)
#     +  {
#     +      return true;
#     +  }
#
# Build each into a separate directory and point M_B_VARIANT_BIN at the resulting binary; this
# script picks it up automatically as a third arm. Without it, the script runs the no-rebuild proxy
# (unified_hash @ mt=1 vs parallel_hash @ mt=1), whose caveats are spelled out in SPEC_HIGH.md.
#
# Env: ARM=baseline|uhj|variant (default: all available), REPEATS, PERF_SECONDS, SKIP_PERF=1,
#      M_B_VARIANT_BIN=/path/to/clickhouse-uhj-b1.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ID=B
OUT="$(m_out_dir "${ID}")"
REPEATS="${REPEATS:-3}"
PERF_SECONDS="${PERF_SECONDS:-30}"
PERF_CARD=16000000
WANT_ARM="${ARM:-}"
START_TS="$(now_epoch)"
VARIANT_BIN="${M_B_VARIANT_BIN:-}"

m_take_lock
echo "== m_${ID}: BITS_FOR_BUCKET_SERIAL / serial map layout =="
echo "# out=${OUT} repeats=${REPEATS}"
if [ -n "${VARIANT_BIN}" ] && [ -x "${VARIANT_BIN}" ]; then
    echo "# rebuild variant present: ${VARIANT_BIN}"
else
    echo "# no rebuild variant (M_B_VARIANT_BIN unset) - running the cross-binary proxy only"
    VARIANT_BIN=""
fi

# `variant` is an extra arm served by a patched UHJ binary. arm_bin/algo dispatch is extended here
# rather than in _common.sh because only this measurement has such an arm.
arm_bin_b() {
    case "$1" in
        variant) echo "${VARIANT_BIN}" ;;
        *) arm_bin "$1" ;;
    esac
}
start_server_b() {
    if [ "$1" = variant ]; then
        CUR_ARM=variant
        CUR_BIN="${VARIANT_BIN}"
        stop_server
        write_server_conf variant
        local cg helper dir="${SRVROOT}/variant"
        cg="$("${WRAP}" --print-cg | awk -F= '/^cg=/{print $2}')"
        helper="${dir}/start_in_cgroup.sh"
        printf '#!/bin/bash\necho $$ | sudo tee %s/cgroup.procs >/dev/null\nexec "%s" server --config-file="%s/config.xml"\n' \
            "${cg}" "${CUR_BIN}" "${dir}" > "${helper}"
        chmod +x "${helper}"
        nohup "${helper}" >"${dir}/log/boot.log" 2>&1 &
        local i
        for i in $(seq 1 180); do server_alive && break; sleep 1; done
        server_alive || { echo "variant server did not start" >&2; exit 1; }
        echo "# server up: arm=variant bin=$(basename "${CUR_BIN}")"
    else
        start_server "$1"
    fi
}

## ---------------------------------------------------------------------------------------------
## Shapes. Everything runs at max_threads=1, where the divergence lives; the probe side is one row
## so the measurement is of the build, and the cardinality sweep is a PK range filter on a
## PK-ordered table, which is exact and cheap.
##
##   u64      all-distinct UInt64 keys. WORST CASE at high cardinality: the flat table's buffer
##            crosses several power-of-two resizes, and each one allocates a buffer twice the size
##            of the live one and rehashes every key in a single pass over a region far larger than
##            the last-level cache. The 256-bucket layout does the same total work in 256 pieces.
##   str48    48-byte String keys via JoinHashMapWithSavedHash: the saved hash makes a resize a pure
##            copy with no re-hashing, which separates "resize copies memory" from "resize rehashes
##            keys" - if the gap persists here it is layout and locality, not hashing.
## ---------------------------------------------------------------------------------------------
sql_u64() {   # sql_u64 <cardinality>
    echo "SELECT count() FROM ${SYNTH_DB}.probe_one AS p
          LEFT JOIN (SELECT id AS k FROM ${SYNTH_DB}.build_u64 WHERE id < $1) AS r ON p.id = r.k"
}
sql_str48() {
    echo "SELECT count() FROM ${SYNTH_DB}.probe_one AS p
          LEFT JOIN (SELECT k FROM ${SYNTH_DB}.build_str48 WHERE id < $1) AS r ON toString(p.id) = r.k"
}
sql_probe_u64() {   # sql_probe_u64 <cardinality> - build AND probe, to see the lookup side too
    echo "SELECT count() FROM ${SYNTH_DB}.probe_10m AS p
          INNER JOIN (SELECT id AS k FROM ${SYNTH_DB}.build_u64 WHERE id < $1) AS r ON p.k = r.k"
}

arm_algo() {
    case "$1" in
        baseline) echo "hash parallel_hash" ;;
        uhj)      echo "unified_hash" ;;
        variant)  echo "unified_hash" ;;
    esac
}

enable_counters() {
    if perfev_available; then MEASURE_SETTINGS=("${PERFEV_SETTINGS[@]}"); echo "# per-query hardware counters: available"
    else MEASURE_SETTINGS=(); echo "# per-query hardware counters: NOT available"; fi
}

## ---------------------------------------------------------------------------------------------
## Pass 1 - cardinality sweep at max_threads=1. The gap should be ~0 while the whole table fits in
## cache and should open up as the buffer outgrows the last-level cache and starts resizing.
## ---------------------------------------------------------------------------------------------
SWEEP="${OUT}/sweep.tsv"

run_sweep_for_arm() {
    local arm="$1" algo card shape sql tag best times maptype flag
    start_server_b "${arm}"
    ensure_synth
    enable_counters
    for algo in $(arm_algo "${arm}"); do
        flag="$(algo_flag "${algo}")"
        for shape in u64 str48 probe_u64; do
            for card in 100000 1000000 4000000 16000000 64000000; do
                # build_str48 only has 16 M rows; skip the two larger points for it.
                if [ "${shape}" = str48 ] && [ "${card}" -gt 16000000 ]; then continue; fi
                case "${shape}" in
                    u64)       sql="$(sql_u64 "${card}")" ;;
                    str48)     sql="$(sql_str48 "${card}")" ;;
                    probe_u64) sql="$(sql_probe_u64 "${card}")" ;;
                esac
                tag="mB_${arm}_${algo}_${shape}_c${card}_mt1"
                tsv_prune "${SWEEP}" "${tag}"
                q_warm "${SYNTH_DB}" "${sql}" "${flag}" --max_threads=1
                maptype="$(q_maptype "${SYNTH_DB}" "${sql}" "${flag}" --max_threads=1)"
                read -r best times <<<"$(q_best "${REPEATS}" "${tag}" "${SYNTH_DB}" "${sql}" "${flag}" --max_threads=1)"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${arm}" "${algo}" "${shape}" "${card}" "${maptype}" "${best}" "${times}" "${tag}" >> "${SWEEP}"
                echo "  ${arm}/${algo} ${shape} card=${card} map=${maptype} best=${best}s"
            done
        done
    done
    qlog_agg "mB_" "${START_TS}" "${OUT}/qlog_${arm}.tsv"
    pplog_dump "mB_" "${START_TS}" "${OUT}/pplog_${arm}.tsv"
    stop_server
}

## ---------------------------------------------------------------------------------------------
## Pass 2 - the worst-case corner under `perf stat`, reproducing the four counters that were
## originally reported for B (instructions, LL misses, dTLB walks, cycles).
## ---------------------------------------------------------------------------------------------
PERFTSV="${OUT}/perf_corners.tsv"
LOOP_SQL=""
LOOP_ARGS=()
loop_once() { q_warm "${SYNTH_DB}" "${LOOP_SQL}" "${LOOP_ARGS[@]}"; }

run_perf_for_arm() {
    local arm="$1" algo prefix iters instr cycles llc dtlb l1d
    start_server_b "${arm}"
    MEASURE_SETTINGS=()
    LOOP_SQL="$(sql_u64 "${PERF_CARD}")"
    tsv_prune_field "${PERFTSV}" 1 "${arm}"
    for algo in $(arm_algo "${arm}"); do
        LOOP_ARGS=("$(algo_flag "${algo}")" --max_threads=1)
        prefix="${OUT}/perf_${arm}_${algo}"
        q_warm "${SYNTH_DB}" "${LOOP_SQL}" "${LOOP_ARGS[@]}"
        echo "  perf ${arm}/${algo} mt=1: core group"
        iters="$(perf_window "${prefix}_core" "${PERF_CORE}" "${PERF_SECONDS}" loop_once | sed 's/iters=//')"
        instr="$(perf_value "${prefix}_core" inst_retired)"
        cycles="$(perf_value "${prefix}_core" cpu_cycles)"
        echo "  perf ${arm}/${algo} mt=1: mem group"
        perf_window "${prefix}_mem" "${PERF_MEM}" "${PERF_SECONDS}" loop_once >/dev/null
        llc="$(perf_value "${prefix}_mem" ll_cache_miss_rd)"
        dtlb="$(perf_value "${prefix}_mem" dtlb_walk)"
        l1d="$(perf_value "${prefix}_mem" l1d_cache_refill)"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${arm}" "${algo}" "${iters}" "${instr}" "${cycles}" "${llc}" "${dtlb}" "${l1d}" "${PERF_CARD}" >> "${PERFTSV}"
        awk -v a="${arm}/${algo}" -v i="${iters}" -v n="${instr}" -v c="${cycles}" -v r="${PERF_CARD}" \
            'BEGIN { if (i > 0) printf "    -> %s: %.1f instr/key, %.1f cyc/key over %d iters\n", a, n/(i*r), c/(i*r), i }'
    done
    stop_server
}

## ---------------------------------------------------------------------------------------------
## Pass 3 - where regime B is reachable in the real world.
##
## Serial UHJ happens when max_threads == 1 (rare in a benchmark) and, much more importantly, for
## every GraceHashJoin in-memory join, which `GraceHashJoin.cpp:739-750` pins to max_threads = 1
## (divergence D18). So the real-world exposure of B is: suite queries run at max_threads=1, plus
## any query that spills. This pass measures the four suites at max_threads=1 on both arms.
## ---------------------------------------------------------------------------------------------
run_suites_mt1() {
    local arm="$1" algo
    start_server_b "${arm}"
    enable_counters
    Q_TIMEOUT="${CENSUS_TIMEOUT:-180}"
    # baseline contributes both layouts: `hash` (flat map, what UHJ's serial map is structurally
    # identical to) and `parallel_hash` at mt=1 (256 sub-tables, the layout being compared against).
    for algo in $(arm_algo "${arm}"); do
        echo "# suites at max_threads=1: ${arm}/${algo}"
        census_exec "mBs_${arm}_$(algo_code "${algo}")" "$(algo_flag "${algo}")" --max_threads=1
    done
    Q_TIMEOUT=0
    stop_server
}

## ---------------------------------------------------------------------------------------------
main() {
    [ -s "${SWEEP}" ]   || printf 'arm\talgo\tshape\tcardinality\tmap_type\tbest_sec\tall_sec\ttag\n' > "${SWEEP}"
    [ -s "${PERFTSV}" ] || printf 'arm\talgo\titers\tinst_retired\tcpu_cycles\tll_cache_miss_rd\tdtlb_walk\tl1d_cache_refill\tkeys\n' > "${PERFTSV}"

    hr; echo "PASS 1: cardinality sweep at max_threads=1"; hr
    if want_arm baseline; then run_sweep_for_arm baseline; fi
    if want_arm uhj;      then run_sweep_for_arm uhj;      fi
    if [ -n "${VARIANT_BIN}" ] && want_arm variant; then run_sweep_for_arm variant; fi

    if [ "${SKIP_PERF:-0}" != 1 ]; then
        hr; echo "PASS 2: perf stat at the worst-case corner (u64, ${PERF_CARD} distinct keys, mt=1)"; hr
        if want_arm baseline; then run_perf_for_arm baseline; fi
        if want_arm uhj;      then run_perf_for_arm uhj;      fi
        if [ -n "${VARIANT_BIN}" ] && want_arm variant; then run_perf_for_arm variant; fi
    fi

    if [ "${SKIP_SUITES:-0}" != 1 ]; then
        hr; echo "PASS 3: four suites at max_threads=1 (where regime B is reachable)"; hr
        if want_arm baseline; then run_suites_mt1 baseline; fi
        if want_arm uhj;      then run_suites_mt1 uhj;      fi
        start_server uhj
        flush_logs
        client --query "
            SELECT splitByChar('_', query_id)[2]                        AS arm,
                   splitByChar('_', query_id)[3]                        AS algo,
                   splitByChar('_', query_id)[4]                        AS suite,
                   count()                                              AS queries,
                   round(sum(query_duration_ms) / 1000, 3)              AS total_sec,
                   round(exp(avg(log(greatest(query_duration_ms, 1)))) / 1000, 4) AS geomean_sec
            FROM system.query_log
            WHERE type = 'QueryFinish' AND query_id LIKE 'mBs\\_%' AND event_time >= toDateTime(${START_TS})
            GROUP BY arm, algo, suite ORDER BY suite, arm, algo
            FORMAT TSVWithNames" > "${OUT}/suites_mt1.tsv" 2>&1 || true
        stop_server
    fi

    hr; echo "SUMMARY m_${ID} - serial map layout"; hr
    echo "-- best-of-${REPEATS} wall time at max_threads=1 (s) --"
    awk -F'\t' 'NR > 1 { printf "%-10s %10s  %-22s %-28s %s\n", $3, $4, $1 "/" $2, $5, $6 }' "${SWEEP}" \
        | sort -k1,1 -k2,2n
    local f
    for f in baseline uhj variant; do
        [ -s "${OUT}/qlog_${f}.tsv" ] || continue
        echo
        echo "-- ${f}: instructions and cache misses per build key --"
        awk -F'\t' 'NR > 1 { printf "%-52s instr/row=%-10s cycles=%-14s llc=%-12s dtlb=%s\n", $1, $12, $8, $9, $10 }' \
            "${OUT}/qlog_${f}.tsv"
    done
    if [ "$(wc -l < "${PERFTSV}")" -gt 1 ]; then
        echo
        echo "-- server-wide perf, u64 / ${PERF_CARD} keys / mt=1, per build key --"
        awk -F'\t' 'NR > 1 && $3 > 0 { printf "%-9s %-14s %9.1f instr %9.1f cyc %8.4f llc %8.5f dtlb %8.4f l1d\n",
                                       $1, $2, $4/($3*$9), $5/($3*$9), $6/($3*$9), $7/($3*$9), $8/($3*$9) }' "${PERFTSV}"
    fi
    if [ -s "${OUT}/suites_mt1.tsv" ]; then
        echo
        echo "-- four suites at max_threads=1 --"
        column -t -s $'\t' "${OUT}/suites_mt1.tsv" || cat "${OUT}/suites_mt1.tsv"
    fi
    hr
    echo "artifacts: ${OUT}"
    echo "M_${ID}_DONE"
}

main "$@"
