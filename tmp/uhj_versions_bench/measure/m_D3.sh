#!/usr/bin/env bash
# D3 - the UHJ parallel build serialises on one global `blocks_mutex`.
#
# Mechanism. Every build thread takes the join-wide `blocks_mutex` at least twice per build block:
# once to register the stored block (`UnifiedHashJoin/HashJoin.cpp:907`) and once inside
# `shrinkStoredBlocksToFit`, which takes the lock BEFORE deciding it has nothing to do
# (`UnifiedHashJoin/HashJoin.cpp:1101`, reached unconditionally from line 1093); RIGHT/FULL joins and
# nullable keys add more (`1045`, `1054`, `1062`), and with `max_bytes_before_external_join` set
# `SpillingHashJoin::addBlockToJoin` adds a third by calling `getTotalByteCount`, which also takes it
# (`SpillingHashJoin.cpp:158`, `UnifiedHashJoin/HashJoin.cpp:701`). `ConcurrentHashJoin` has no
# join-wide lock at all: each shard has its own mutex taken with `try_to_lock`
# (`ConcurrentHashJoin.cpp:340`) and the global totals are relaxed atomics.
#
# Needs no rebuild. The cost is PER BLOCK, and every other build-side divergence between the arms -
# the scatter of divergence A above all - is PER ROW. So holding the row count fixed and shrinking
# `max_block_size` multiplies the number of critical sections while leaving the per-row work
# untouched: the SLOPE of build time against block count is D3, and the intercept absorbs A. That is
# the whole design; the thread sweep on top of it shows where the ceiling starts to bind.
#
# Env: ARM=baseline|uhj (default: both), REPEATS, ROWS, PERF_SECONDS, SKIP_SPILLCHECK=1,
#      SKIP_PERF=1, SKIP_SUITES=1.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ID=D3
OUT="$(m_out_dir "${ID}")"
REPEATS="${REPEATS:-3}"
ROWS="${ROWS:-16000000}"
PERF_SECONDS="${PERF_SECONDS:-25}"
WANT_ARM="${ARM:-}"
START_TS="$(now_epoch)"

# Large enough that the SpillingHashJoin wrapper exists and calls getTotalByteCount once per block
# without the join ever actually switching to GraceHashJoin.
NOSPILL_CAP="${NOSPILL_CAP:-68719476736}"   # 64 GiB

BLOCK_SIZES="65536 8192 2048 512"

m_take_lock
echo "== m_${ID}: global blocks_mutex on the parallel build =="
echo "# out=${OUT} rows=${ROWS} repeats=${REPEATS}"

## ---------------------------------------------------------------------------------------------
## Shapes. Build-only (`probe_one` has one row), so what is measured is the build phase.
##
##   u64    one UInt64 key, no payload. The cheapest possible per-row work, which maximises the
##          share of the build that is lock traffic. Two `blocks_mutex` acquisitions per block.
##   right  a Nullable key joined RIGHT, so `isRightOrFull(kind)` holds and the nullmap is stored:
##          a THIRD mandatory acquisition per block (line 1054, or line 1045 if the kind ends up
##          using per-row flags). WORST CASE for D3 - 50% more global critical sections per block
##          for identical per-row work.
##
## `max_block_size` sets the number of build blocks. `preferred_block_size_bytes=0` switches off the
## adaptive sizing that would otherwise override it for narrow rows, so the requested block size is
## the delivered one.
## ---------------------------------------------------------------------------------------------
BLOCK_ARGS=(--preferred_block_size_bytes=0)

sql_u64() {
    echo "SELECT count() FROM ${SYNTH_DB}.probe_one AS p
          LEFT JOIN (SELECT id AS k FROM ${SYNTH_DB}.build_u64 WHERE id < ${ROWS}) AS r ON p.id = r.k"
}
sql_right() {
    echo "SELECT count() FROM ${SYNTH_DB}.probe_one AS p
          RIGHT JOIN (SELECT k FROM ${SYNTH_DB}.build_u64_null WHERE id < ${ROWS}) AS r ON p.id = r.k"
}
shape_sql() { case "$1" in u64) sql_u64 ;; right) sql_right ;; esac; }

# build_u64_null holds 16 M rows, so the `right` shape saturates there.
shape_rows() {
    case "$1" in
        u64)   echo "${ROWS}" ;;
        right) if [ "${ROWS}" -lt 16000000 ]; then echo "${ROWS}"; else echo 16000000; fi ;;
    esac
}
# The thread axis is what the `u64` shape is for; `right` only has to show that an extra critical
# section per block costs extra, which the two ends of the thread range already answer.
shape_threads() { case "$1" in u64) echo "1 2 4 8 16" ;; right) echo "1 16" ;; esac; }

arm_algos() {
    case "$1" in
        baseline) echo "parallel_hash" ;;
        uhj)      echo "unified_hash" ;;
    esac
}

enable_counters() {
    if perfev_available; then MEASURE_SETTINGS=("${PERFEV_SETTINGS[@]}"); echo "# per-query hardware counters: available"
    else MEASURE_SETTINGS=(); echo "# per-query hardware counters: NOT available"; fi
}

## ---------------------------------------------------------------------------------------------
## Pass 1 - threads x block size, at a fixed row count.
##
## Reading the result:
##   * across max_threads at a fixed block size -> the scalability curve. UHJ's mt=1 point is also
##     where divergence B lives (one-bucket map), so compare curve SHAPES, and normalise against
##     mt=2 rather than mt=1.
##   * across block size at a fixed max_threads -> the per-block cost. The row count is constant, so
##     whatever time appears as the blocks get smaller is per-block cost, and the arm-to-arm
##     difference in that slope is D3.
## ---------------------------------------------------------------------------------------------
GRID="${OUT}/grid.tsv"

run_grid_for_arm() {
    local arm="$1" algo code shape bs mt sql rows tag best times streams build_us flag
    start_server "${arm}"
    ensure_synth
    enable_counters
    for algo in $(arm_algos "${arm}"); do
        flag="$(algo_flag "${algo}")"
        code="$(algo_code "${algo}")"
        for shape in u64 right; do
            sql="$(shape_sql "${shape}")"
            rows="$(shape_rows "${shape}")"
            for bs in ${BLOCK_SIZES}; do
                for mt in $(shape_threads "${shape}"); do
                    tag="mD3g_${code}_${shape}_bs${bs}_mt${mt}"
                    tsv_prune "${GRID}" "${tag}"
                    q_warm "${SYNTH_DB}" "${sql}" "${flag}" "--max_threads=${mt}" \
                        "--max_block_size=${bs}" "${BLOCK_ARGS[@]}"
                    streams="$(q_build_streams "${SYNTH_DB}" "${sql}" "${flag}" "--max_threads=${mt}")"
                    read -r best times <<<"$(q_best "${REPEATS}" "${tag}" "${SYNTH_DB}" "${sql}" \
                        "${flag}" "--max_threads=${mt}" "--max_block_size=${bs}" "${BLOCK_ARGS[@]}")"
                    build_us="$(pp_build_us "${tag}_r1")"
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "${arm}" "${code}" "${shape}" "${bs}" "${mt}" "${streams}" \
                        "${best}" "${times}" "${build_us}" "$((rows / bs))" "${tag}" >> "${GRID}"
                    echo "  ${code} ${shape} bs=${bs} mt=${mt} streams=${streams} best=${best}s build_us=${build_us}"
                done
            done
        done
    done
    qlog_agg "mD3g_" "${START_TS}" "${OUT}/qlog_${arm}.tsv"
    pplog_dump "mD3g_" "${START_TS}" "${OUT}/pplog_${arm}.tsv"
    stop_server
}

## ---------------------------------------------------------------------------------------------
## Pass 2 - the extra acquisition that the spilling wrapper adds.
##
## With `max_bytes_before_external_join` set to a value large enough never to spill, both arms are
## wrapped in `SpillingHashJoin`, which calls `getTotalByteCount()` once per build block. On the
## baseline that walks ConcurrentHashJoin's atomics; on UHJ it takes `blocks_mutex` a third time.
## The quantity of interest is the WITHIN-ARM ratio (capped / uncapped), because enabling the
## wrapper also makes both arms save the key columns, which changes the stored bytes on both.
## ---------------------------------------------------------------------------------------------
SPILLCHK="${OUT}/spillcheck.tsv"

run_spillcheck_for_arm() {
    local arm="$1" algo code bs mode sql tag best times cap flag
    start_server "${arm}"
    ensure_synth
    enable_counters
    sql="$(sql_u64)"
    for algo in $(arm_algos "${arm}"); do
        flag="$(algo_flag "${algo}")"
        code="$(algo_code "${algo}")"
        for bs in 65536 512; do
            for mode in off on; do
                cap=0
                [ "${mode}" = on ] && cap="${NOSPILL_CAP}"
                tag="mD3c_${code}_bs${bs}_${mode}"
                tsv_prune "${SPILLCHK}" "${tag}"
                q_warm "${SYNTH_DB}" "${sql}" "${flag}" --max_threads=16 \
                    "--max_block_size=${bs}" "--max_bytes_before_external_join=${cap}" "${BLOCK_ARGS[@]}"
                read -r best times <<<"$(q_best "${REPEATS}" "${tag}" "${SYNTH_DB}" "${sql}" \
                    "${flag}" --max_threads=16 "--max_block_size=${bs}" \
                    "--max_bytes_before_external_join=${cap}" "${BLOCK_ARGS[@]}")"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${arm}" "${code}" "${bs}" "${mode}" "${best}" "${times}" \
                    "$((ROWS / bs))" "${tag}" >> "${SPILLCHK}"
                echo "  ${code} bs=${bs} spilling_wrapper=${mode} best=${best}s"
            done
        done
    done
    stop_server
}

## ---------------------------------------------------------------------------------------------
## Pass 3 - direct contention counters.
##
## An uncontended `std::mutex` costs two atomics and issues no syscall; a contended one parks the
## loser in `futex`. `insertIntoSlots` additionally calls `sched_yield` when it cannot acquire any
## slot lock. Counting both tracepoints server-wide over a fixed window and dividing by the number
## of build blocks processed in that window turns "is the lock contended" into a number. If the
## tracepoints are not permitted the counters come back as zero and pass 1's slope stands alone.
## ---------------------------------------------------------------------------------------------
PERFTSV="${OUT}/perf_lock.tsv"
LOOP_SQL=""
LOOP_ARGS=()
loop_once() { q_warm "${SYNTH_DB}" "${LOOP_SQL}" "${LOOP_ARGS[@]}"; }

run_perf_for_arm() {
    local arm="$1" algo code bs prefix iters futex yields ctxsw instr cycles
    start_server "${arm}"
    MEASURE_SETTINGS=()
    LOOP_SQL="$(sql_u64)"
    for algo in $(arm_algos "${arm}"); do
        code="$(algo_code "${algo}")"
        for bs in 65536 512; do
            LOOP_ARGS=("$(algo_flag "${algo}")" --max_threads=16 "--max_block_size=${bs}" "${BLOCK_ARGS[@]}")
            prefix="${OUT}/perf_${code}_bs${bs}"
            q_warm "${SYNTH_DB}" "${LOOP_SQL}" "${LOOP_ARGS[@]}"
            echo "  perf ${code} bs=${bs}: lock group"
            iters="$(perf_window "${prefix}_lock" "${PERF_LOCK},context-switches" "${PERF_SECONDS}" loop_once | sed 's/iters=//')"
            futex="$(perf_value "${prefix}_lock" sys_enter_futex)"
            yields="$(perf_value "${prefix}_lock" sys_enter_sched_yield)"
            ctxsw="$(perf_value "${prefix}_lock" context-switches)"
            echo "  perf ${code} bs=${bs}: core group"
            perf_window "${prefix}_core" "${PERF_CORE}" "${PERF_SECONDS}" loop_once >/dev/null
            instr="$(perf_value "${prefix}_core" inst_retired)"
            cycles="$(perf_value "${prefix}_core" cpu_cycles)"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${arm}" "${code}" "${bs}" "${iters}" "${futex}" "${yields}" \
                "${ctxsw}" "${instr}" "${cycles}" "$((ROWS / bs))" >> "${PERFTSV}"
            awk -v a="${code}" -v b="${bs}" -v i="${iters}" -v f="${futex}" -v y="${yields}" \
                -v n="$((ROWS / bs))" \
                'BEGIN { if (i > 0) printf "    -> %s bs=%s: %.2f futex/block, %.2f yield/block\n", a, b, f/(i*n), y/(i*n) }'
        done
    done
    stop_server
}

## ---------------------------------------------------------------------------------------------
## Pass 4 - real-world exposure.
##
## Exposure is (number of build blocks) x (number of build threads), and the per-query proxy for the
## first factor is `JoinBuildTableRowCount / max_block_size`. The suites are executed once per arm at
## max_threads=16 and the build-side processor time is read back from
## `system.processors_profile_log`: `FillingRightJoinSide.elapsed_us` summed over streams includes
## the time a thread spends blocked on `blocks_mutex`, so the arm-to-arm ratio of that number, at
## equal build rows, is what D3 looks like in a real query.
## ---------------------------------------------------------------------------------------------
suite_args() {
    case "$1" in
        baseline) echo "--join_algorithm=direct,parallel_hash,hash --max_threads=16" ;;
        uhj)      echo "--join_algorithm=unified_hash --max_threads=16" ;;
    esac
}
suite_code() { case "$1" in baseline) echo b ;; uhj) echo u ;; esac; }

run_suites_for_arm() {
    local arm="$1"
    local -a args
    start_server "${arm}"
    enable_counters
    # suite_args returns several flags in one string; splitting them is the point.
    # shellcheck disable=SC2206
    args=( $(suite_args "${arm}") )
    Q_TIMEOUT="${CENSUS_TIMEOUT:-300}"
    census_exec "mD3s$(suite_code "${arm}")" "${args[@]}"
    Q_TIMEOUT=0
    stop_server
}

summarize_suites() {
    local raw="${OUT}/suites_build.tsv" body="${OUT}/.suites_ratio_body"
    start_server uhj
    flush_logs
    client --query "
        WITH build AS (
            SELECT query_id, sum(elapsed_us) AS build_us
            FROM system.processors_profile_log
            WHERE startsWith(query_id, 'mD3s') AND event_time >= toDateTime(${START_TS})
              AND name = 'FillingRightJoinSide'
            GROUP BY query_id
        )
        SELECT splitByChar('_', q.query_id)[2]                              AS suite,
               splitByChar('_', q.query_id)[3]                              AS q,
               substring(splitByChar('_', q.query_id)[1], 5, 1)             AS arm,
               q.ProfileEvents['JoinBuildTableRowCount']                    AS build_rows,
               build.build_us                                               AS build_us,
               round(q.query_duration_ms / 1000, 4)                         AS sec
        FROM system.query_log AS q
        INNER JOIN build ON build.query_id = q.query_id
        WHERE q.type = 'QueryFinish'
          AND startsWith(q.query_id, 'mD3s')
          AND q.event_time >= toDateTime(${START_TS})
        ORDER BY suite, q, arm
        FORMAT TSVWithNames" > "${raw}" 2>&1 || true

    # The comparison itself: same query, same build rows, build-side processor time on each arm.
    awk -F'\t' 'NR > 1 {
            k = $1 "\t" $2
            if ($3 == "b") { br[k] = $4; bu[k] = $5 } else { ur[k] = $4; uu[k] = $5 }
        }
        END {
            for (k in bu)
                if (k in uu && bu[k] > 0)
                    printf "%s\t%d\t%d\t%d\t%.2f\n", k, br[k], bu[k], uu[k], uu[k] / bu[k]
        }' "${raw}" | sort -t$'\t' -k6,6gr > "${body}"
    {
        printf 'suite\tq\tbuild_rows\tbaseline_build_us\tuhj_build_us\tratio\n'
        cat "${body}"
    } > "${OUT}/suites_ratio.tsv"
    rm -f "${body}"
    stop_server
}

## ---------------------------------------------------------------------------------------------
## Summary helpers. awk here is POSIX awk, so composite keys go through SUBSEP rather than the
## gawk-only nested-array syntax.
## ---------------------------------------------------------------------------------------------
print_wall_table() {
    printf '   %-4s %-6s %-4s %10s %10s %10s %10s\n' algo shape mt bs65536 bs8192 bs2048 bs512
    awk -F'\t' '
        NR > 1 { k = $2 SUBSEP $3 SUBSEP $5; t[k SUBSEP $4] = $7; seen[k] = 1 }
        END {
            for (k in seen) {
                split(k, a, SUBSEP)
                printf "   %-4s %-6s %-4s %10s %10s %10s %10s\n", a[1], a[2], a[3],
                       t[k SUBSEP 65536], t[k SUBSEP 8192], t[k SUBSEP 2048], t[k SUBSEP 512]
            }
        }' "${GRID}" | sort -k1,1 -k2,2 -k3,3n
}

print_per_block_table() {
    printf '   %-4s %-6s %-4s %14s %14s\n' algo shape mt us_per_block us_per_block_build
    awk -F'\t' '
        NR > 1 {
            k = $2 SUBSEP $3 SUBSEP $5
            t[k SUBSEP $4] = $7; b[k SUBSEP $4] = $9; n[k SUBSEP $4] = $10; seen[k] = 1
        }
        END {
            for (k in seen) {
                hi = k SUBSEP 512; lo = k SUBSEP 65536
                if (!(hi in t) || !(lo in t)) continue
                dn = n[hi] - n[lo]
                if (dn <= 0) continue
                split(k, a, SUBSEP)
                printf "   %-4s %-6s %-4s %14.3f %14.3f\n", a[1], a[2], a[3],
                       1e6 * (t[hi] - t[lo]) / dn, (b[hi] - b[lo]) / dn
            }
        }' "${GRID}" | sort -k1,1 -k2,2 -k3,3n
}

print_spillcheck_table() {
    printf '   %-4s %-8s %10s %10s %10s\n' algo bs off on on/off
    awk -F'\t' '
        NR > 1 { v[$2 SUBSEP $3 SUBSEP $4] = $5; seen[$2 SUBSEP $3] = 1 }
        END {
            for (k in seen) {
                split(k, a, SUBSEP)
                off = v[k SUBSEP "off"]; on = v[k SUBSEP "on"]
                if (off + 0 <= 0) continue
                printf "   %-4s %-8s %10s %10s %10.3f\n", a[1], a[2], off, on, on / off
            }
        }' "${SPILLCHK}" | sort -k1,1 -k2,2n
}

## ---------------------------------------------------------------------------------------------
main() {
    [ -s "${GRID}" ] || printf 'arm\talgo\tshape\tmax_block_size\tmax_threads\tbuild_streams\tbest_sec\tall_sec\tbuild_us\tnominal_blocks\ttag\n' > "${GRID}"
    [ -s "${SPILLCHK}" ] || printf 'arm\talgo\tmax_block_size\tspilling_wrapper\tbest_sec\tall_sec\tnominal_blocks\ttag\n' > "${SPILLCHK}"
    [ -s "${PERFTSV}" ] || printf 'arm\talgo\tmax_block_size\titers\tfutex\tsched_yield\tcontext_switches\tinst_retired\tcpu_cycles\tnominal_blocks\n' > "${PERFTSV}"

    hr; echo "PASS 1: max_threads x max_block_size at ${ROWS} build rows"; hr
    if want_arm baseline; then run_grid_for_arm baseline; fi
    if want_arm uhj;      then run_grid_for_arm uhj;      fi

    if [ "${SKIP_SPILLCHECK:-0}" != 1 ]; then
        hr; echo "PASS 2: the getTotalByteCount acquisition the SpillingHashJoin wrapper adds"; hr
        if want_arm baseline; then run_spillcheck_for_arm baseline; fi
        if want_arm uhj;      then run_spillcheck_for_arm uhj;      fi
    fi

    if [ "${SKIP_PERF:-0}" != 1 ]; then
        hr; echo "PASS 3: futex / sched_yield per build block (server-wide perf, mt=16)"; hr
        if want_arm baseline; then run_perf_for_arm baseline; fi
        if want_arm uhj;      then run_perf_for_arm uhj;      fi
    fi

    if [ "${SKIP_SUITES:-0}" != 1 ]; then
        hr; echo "PASS 4: four suites at max_threads=16, build-side processor time"; hr
        if want_arm baseline; then run_suites_for_arm baseline; fi
        if want_arm uhj;      then run_suites_for_arm uhj;      fi
        summarize_suites
    fi

    hr; echo "SUMMARY m_${ID} - global blocks_mutex"; hr
    echo "-- best-of-${REPEATS} build-only wall time (s); rows are fixed, so the columns differ only"
    echo "   in how many build blocks those rows arrive in --"
    print_wall_table
    echo
    echo "-- per-block cost, from the two ends of the block-size axis:"
    echo "   (t[bs=512] - t[bs=65536]) / (blocks[512] - blocks[65536]) --"
    print_per_block_table
    if [ "$(wc -l < "${SPILLCHK}")" -gt 1 ]; then
        echo
        echo "-- wall time with and without the SpillingHashJoin wrapper's per-block"
        echo "   getTotalByteCount call (mt=16, never actually spills) --"
        print_spillcheck_table
    fi
    if [ "$(wc -l < "${PERFTSV}")" -gt 1 ]; then
        echo
        echo "-- lock traffic per build block (mt=16) --"
        awk -F'\t' 'NR > 1 && $4 > 0 && $10 > 0 {
                printf "   %-4s bs=%-6s %9.3f futex %9.3f yield %9.3f ctxsw %9.1f instr\n",
                       $2, $3, $5/($4*$10), $6/($4*$10), $7/($4*$10), $8/($4*$10) }' "${PERFTSV}"
    fi
    if [ -s "${OUT}/suites_ratio.tsv" ]; then
        echo
        echo "-- real world: top 20 suite queries by uhj/baseline build-side processor time --"
        head -21 "${OUT}/suites_ratio.tsv" | column -t -s $'\t' || head -21 "${OUT}/suites_ratio.tsv"
    fi
    hr
    echo "artifacts: ${OUT}"
    echo "M_${ID}_DONE"
}

main "$@"
