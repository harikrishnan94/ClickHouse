#!/usr/bin/env bash
# D2 - `bucket_bytes` under-reports the hash table while the build is running.
#
# Mechanism. `insertIntoSlots` samples `slot_bytes(slot)` before and after each insert and adds only
# the difference to `data->bucket_bytes` (`UnifiedHashJoin/HashJoin.cpp:140,166,222`), and the map's
# buffers are allocated by `MapsTemplate::create` before the first insert, so the initial allocation
# sits inside `bytes_before` and cancels out of every delta - deliberately, see the comment at
# `UnifiedHashJoin/HashJoin.cpp:438-439`. Baseline `HashJoin::getTotalByteCount`
# (`HashJoin/HashJoin.cpp:533-557`) recomputes the map contribution from the maps themselves on
# every call, so it always includes that buffer.
#
# Size of the un-accounted amount, from the source:
#   * two-level map (any `max_threads > 1`): 256 buckets x the grower's initial 256 cells = 65 536
#     cells; a `key64` cell is `HashMapCell<UInt64, RowRefList>` = 8 + 8 bytes, so ~1 MiB. This is
#     the "~1 MiB" the code comment refers to.
#   * one-bucket serial map (`max_threads == 1`): 256 cells, i.e. a few KiB - negligible.
#   * fixed maps (`key8`, `key16`, `range*`): the WHOLE buffer, because a `FixedHashTable` allocates
#     `2^size_bits` cells in its constructor and never grows, so the delta is exactly zero forever.
#     `key16` is 2^16 cells; `range18_key64` is 2^18.
#
# Needs no rebuild. `getTotalByteCount` is observable from SQL in two independent ways:
#
#   (1) `max_bytes_in_join` + `join_overflow_mode=throw`. The check is `bytes > max_bytes`
#       (`QueryPipeline/SizeLimits.cpp:40`) against exactly the value under study, so the SMALLEST
#       limit at which a query still succeeds IS the peak the join reported during its build. A
#       bisection on that limit reads the accounting out to the byte, and the exception text
#       ("current bytes: 1.25 MiB") corroborates it to two decimal digits.
#   (2) `max_bytes_before_external_join`. `SpillingHashJoin::addBlockToJoin` spills when
#       `getTotalByteCount() * 2 >= max_bytes_before_external_join` (`SpillingHashJoin.cpp:153,158`),
#       so under-reporting moves the spill point to a larger build side - which is the consequence a
#       user actually feels, and which `JoinSpillingHashJoinSwitchedToGraceJoin` reports per query.
#
# Pass 1 measures the accounting itself (bytes), pass 2 measures the consequence (spill point and
# peak memory), pass 3 measures how often the suites are exposed to it at all.
#
# Env: ARM=baseline|uhj (default: both), SPILL_CAP (bytes, default 64 MiB), SKIP_SPILL=1,
#      SKIP_SUITES=1.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ID=D2
OUT="$(m_out_dir "${ID}")"
WANT_ARM="${ARM:-}"
START_TS="$(now_epoch)"
SPILL_CAP="${SPILL_CAP:-67108864}"          # 64 MiB
SUITE_CAP="${SUITE_CAP:-536870912}"         # 512 MiB, for the suite pass

m_take_lock
echo "== m_${ID}: bucket_bytes under-reporting =="
echo "# out=${OUT} spill_cap=${SPILL_CAP}"

## ---------------------------------------------------------------------------------------------
## Configurations. `max_threads` is part of the configuration and not a swept axis, because it
## decides the map layout and therefore the size of the gap: mt=1 is a one-bucket map (256 cells
## un-accounted), mt=16 is 256 buckets (65 536 cells un-accounted).
## ---------------------------------------------------------------------------------------------
cfg_args() {
    case "$1" in
        bh)     echo "--join_algorithm=hash --max_threads=1" ;;
        bhmt16) echo "--join_algorithm=hash --max_threads=16" ;;
        bph)    echo "--join_algorithm=parallel_hash --max_threads=16" ;;
        umt1)   echo "--join_algorithm=unified_hash --max_threads=1" ;;
        umt16)  echo "--join_algorithm=unified_hash --max_threads=16" ;;
    esac
}
arm_cfgs() { case "$1" in baseline) echo "bh bhmt16 bph" ;; uhj) echo "umt1 umt16" ;; esac; }

## ---------------------------------------------------------------------------------------------
## Shapes.
##
##   u16fixed  WORST CASE. A UInt16 join key selects `key16`, a `PartitionedFixedHashMap` whose
##             2^16-cell buffer is allocated in the constructor and never grows, so UHJ's delta
##             accounting reports ZERO bytes for the map no matter how many rows are inserted, for
##             the entire build. Only 1 000 build rows, so the stored columns are ~KiB and
##             essentially the whole reported number is the map: the arm-to-arm difference is the
##             un-accounted buffer, undiluted.
##   u64small  100 000 distinct UInt64 keys -> `key64`. The map is a few MiB, so the fixed ~1 MiB
##             initial allocation is a large fraction of it.
##   u64mid    2 000 000 keys. Same absolute gap, ~30x more map, so this is where the RELATIVE
##             error becomes small - included so the sweep shows the gap shrinking, not just its
##             maximum.
##
## All three are build-only (`probe_one` has one row) so nothing on the probe side contributes to
## the numbers.
## ---------------------------------------------------------------------------------------------
shape_rows() { case "$1" in u16fixed) echo 1000 ;; u64small) echo 100000 ;; u64mid) echo 2000000 ;; esac; }
shape_sql() {
    case "$1" in
        u16fixed)
            echo "SELECT count() FROM ${SYNTH_DB}.probe_one AS p
                  LEFT JOIN ${SYNTH_DB}.build_u16 AS r ON toUInt16(p.id) = r.k" ;;
        u64small)
            echo "SELECT count() FROM ${SYNTH_DB}.probe_one AS p
                  LEFT JOIN (SELECT id AS k FROM ${SYNTH_DB}.build_u64 WHERE id < 100000) AS r ON p.id = r.k" ;;
        u64mid)
            echo "SELECT count() FROM ${SYNTH_DB}.probe_one AS p
                  LEFT JOIN (SELECT id AS k FROM ${SYNTH_DB}.build_u64 WHERE id < 2000000) AS r ON p.id = r.k" ;;
    esac
}

## ---------------------------------------------------------------------------------------------
## Reading `getTotalByteCount` out of the server.
## ---------------------------------------------------------------------------------------------

# min_pass_bytes <db> <sql> <args...>
#   The smallest `max_bytes_in_join` at which the query succeeds, i.e. the peak byte count the join
#   reported during its build. Doubling search for an upper bound, then bisection to 4 KiB.
#   Prints 0 when even the largest bound fails - which means the query is failing for some reason
#   other than the join size limit, and the cell must be discarded rather than believed.
min_pass_bytes() {
    local db="$1" sql="$2"; shift 2
    local lo=1 hi=0 cand mid
    for cand in 1048576 4194304 16777216 67108864 268435456 1073741824 4294967296 17179869184; do
        if q_ok "${db}" "${sql}" "$@" --max_bytes_in_join="${cand}" --join_overflow_mode=throw; then
            hi="${cand}"; break
        fi
        lo="${cand}"
    done
    if [ "${hi}" = 0 ]; then echo 0; return; fi
    while [ $((hi - lo)) -gt 4096 ]; do
        mid=$(((lo + hi) / 2))
        if q_ok "${db}" "${sql}" "$@" --max_bytes_in_join="${mid}" --join_overflow_mode=throw; then
            hi="${mid}"
        else
            lo="${mid}"
        fi
    done
    echo "${hi}"
}

# reported_first_block <db> <sql> <args...>
#   `max_bytes_in_join=1` makes the very first limit check fail, and the exception carries the
#   reported total at that moment. An independent corroboration of the bisection that costs one
#   query instead of thirty, and the one number that isolates the state of the accounting at the
#   START of the build, where the un-accounted initial allocation is the whole story.
reported_first_block() {
    local db="$1" sql="$2"; shift 2
    local msg
    msg="$(q_err "${db}" "${sql}" "$@" --max_bytes_in_join=1 --join_overflow_mode=throw)"
    case "${msg}" in
        *"current bytes:"*) readable_to_bytes "${msg#*current bytes: }" ;;
        *) echo 0 ;;
    esac
}

ACC="${OUT}/accounting.tsv"

run_accounting_for_arm() {
    local arm="$1" cfg shape sql rows tag minpass first mem qid
    local -a args
    start_server "${arm}"
    ensure_synth
    MEASURE_SETTINGS=()
    for cfg in $(arm_cfgs "${arm}"); do
        # cfg_args deliberately returns several flags in one string; splitting it is the point.
        # shellcheck disable=SC2206
        args=( $(cfg_args "${cfg}") )
        for shape in u16fixed u64small u64mid; do
            sql="$(shape_sql "${shape}")"
            rows="$(shape_rows "${shape}")"
            tag="mD2a_${cfg}_${shape}"
            tsv_prune "${ACC}" "${tag}"

            # Reference run with no limit at all: the real peak memory of the query, which is what
            # the reported number is supposed to approximate.
            qid="${tag}_ref"
            q_time "${qid}" "${SYNTH_DB}" "${sql}" "${args[@]}" >/dev/null
            mem="$(qlog_field "${qid}" "memory_usage")"

            first="$(reported_first_block "${SYNTH_DB}" "${sql}" "${args[@]}")"
            minpass="$(min_pass_bytes "${SYNTH_DB}" "${sql}" "${args[@]}")"

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${arm}" "${cfg}" "${shape}" "${rows}" "${first}" "${minpass}" "${mem}" "${tag}" >> "${ACC}"
            echo "  ${cfg} ${shape}: reported_after_first_block=${first} peak_reported=${minpass} peak_memory=${mem}"
        done
    done
    stop_server
}

## ---------------------------------------------------------------------------------------------
## Pass 2 - the consequence: where the spill happens.
##
## `max_bytes_before_external_join` is fixed and the build side is swept, so the output is the
## build size at which each arm switches to GraceHashJoin. An arm that under-reports keeps more
## data in memory before switching: it should spill at a LARGER build side and show a HIGHER peak
## memory at the switch. Both are read from `system.query_log`, so no log scraping is needed.
##
## The probe side is `probe_10m` here rather than one row: a spilled join has to re-read and
## re-partition the probe side too, and a one-row probe would hide the part of the cost that the
## delayed spill actually shifts around.
## ---------------------------------------------------------------------------------------------
SPILL="${OUT}/spill.tsv"

sql_spill() {   # sql_spill <build rows>
    echo "SELECT count() FROM ${SYNTH_DB}.probe_10m AS p
          INNER JOIN (SELECT id AS k FROM ${SYNTH_DB}.build_u64 WHERE id < $1) AS r ON p.k = r.k"
}

run_spill_for_arm() {
    local arm="$1" cfg n sql tag qid spilled mem sec rows_built
    local -a args
    start_server "${arm}"
    ensure_synth
    MEASURE_SETTINGS=()
    for cfg in $(arm_cfgs "${arm}"); do
        # shellcheck disable=SC2206
        args=( $(cfg_args "${cfg}") )
        for n in 250000 500000 1000000 2000000 3000000 4000000 6000000; do
            sql="$(sql_spill "${n}")"
            tag="mD2s_${cfg}_n${n}"
            tsv_prune "${SPILL}" "${tag}"
            qid="${tag}_r1"
            q_warm "${SYNTH_DB}" "${sql}" "${args[@]}" "--max_bytes_before_external_join=${SPILL_CAP}"
            sec="$(q_time "${qid}" "${SYNTH_DB}" "${sql}" "${args[@]}" "--max_bytes_before_external_join=${SPILL_CAP}")"
            spilled="$(qlog_field "${qid}" "ProfileEvents['JoinSpillingHashJoinSwitchedToGraceJoin']")"
            mem="$(qlog_field "${qid}" "memory_usage")"
            rows_built="$(qlog_field "${qid}" "ProfileEvents['JoinBuildTableRowCount']")"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${arm}" "${cfg}" "${n}" "${spilled:-0}" "${mem:-0}" "${sec}" "${rows_built:-0}" "${tag}" >> "${SPILL}"
            echo "  ${cfg} build=${n} spilled=${spilled:-0} peak_mem=${mem:-0} ${sec}s"
        done
    done
    stop_server
}

## ---------------------------------------------------------------------------------------------
## Pass 3 - real-world exposure.
##
## D2 is unobservable unless a memory bound is set, so the real-world question is not "which query
## touches this code" but "of the queries that do have a bound, how many land on the wrong side of
## it". The suites are run twice per arm - once unbounded, once with `max_bytes_before_external_join
## = SUITE_CAP` - and the two runs are compared on spill count and total time. A query that spills
## on one arm and not the other is D2 (or D1's parallelism decision) changing the plan's behaviour
## under the same configured bound.
## ---------------------------------------------------------------------------------------------
## The suite pass compares each arm as a user would actually configure it - the baseline arm with
## its shipped `join_algorithm` list, not with `hash` pinned - because the question here is how
## often the bound is hit in practice, not which map layout is responsible.
suite_args() {
    case "$1" in
        baseline) echo "--join_algorithm=direct,parallel_hash,hash --max_threads=16" ;;
        uhj)      echo "--join_algorithm=unified_hash --max_threads=16" ;;
    esac
}
suite_code() { case "$1" in baseline) echo b ;; uhj) echo u ;; esac; }

run_suites_for_arm() {
    local arm="$1" code
    local -a args
    start_server "${arm}"
    MEASURE_SETTINGS=()
    code="$(suite_code "${arm}")"
    # shellcheck disable=SC2206
    args=( $(suite_args "${arm}") )
    Q_TIMEOUT="${CENSUS_TIMEOUT:-300}"
    echo "# suites, ${arm}, unbounded"
    census_exec "mD2n${code}" "${args[@]}" --max_bytes_before_external_join=0
    echo "# suites, ${arm}, cap=${SUITE_CAP}"
    census_exec "mD2c${code}" "${args[@]}" "--max_bytes_before_external_join=${SUITE_CAP}"
    Q_TIMEOUT=0
    stop_server
}

summarize_suites() {
    start_server uhj
    flush_logs
    # query ids are `mD2<n|c><b|u>_<suite>_q<n>_r1`: character 4 is the bound, character 5 the arm.
    client --query "
        SELECT substring(splitByChar('_', query_id)[1], 4, 1)       AS bound,
               substring(splitByChar('_', query_id)[1], 5, 1)       AS arm,
               splitByChar('_', query_id)[2]                        AS suite,
               count()                                              AS queries,
               countIf(ProfileEvents['JoinSpillingHashJoinSwitchedToGraceJoin'] > 0) AS spilled,
               round(sum(query_duration_ms) / 1000, 2)              AS total_sec,
               formatReadableSize(max(memory_usage))                AS peak_mem
        FROM system.query_log
        WHERE type = 'QueryFinish'
          AND (startsWith(query_id, 'mD2n') OR startsWith(query_id, 'mD2c'))
          AND event_time >= toDateTime(${START_TS})
        GROUP BY bound, arm, suite
        ORDER BY suite, bound, arm
        FORMAT TSVWithNames" > "${OUT}/suites.tsv" 2>&1 || true
    client --query "
        SELECT splitByChar('_', query_id)[2]                        AS suite,
               splitByChar('_', query_id)[3]                        AS q,
               maxIf(ProfileEvents['JoinSpillingHashJoinSwitchedToGraceJoin'],
                     substring(splitByChar('_', query_id)[1], 5, 1) = 'b')    AS spilled_baseline,
               maxIf(ProfileEvents['JoinSpillingHashJoinSwitchedToGraceJoin'],
                     substring(splitByChar('_', query_id)[1], 5, 1) = 'u')    AS spilled_uhj,
               maxIf(memory_usage, substring(splitByChar('_', query_id)[1], 5, 1) = 'b') AS mem_baseline,
               maxIf(memory_usage, substring(splitByChar('_', query_id)[1], 5, 1) = 'u') AS mem_uhj
        FROM system.query_log
        WHERE type = 'QueryFinish'
          AND startsWith(query_id, 'mD2c')
          AND event_time >= toDateTime(${START_TS})
        GROUP BY suite, q
        HAVING spilled_baseline != spilled_uhj
        ORDER BY suite, q
        FORMAT TSVWithNames" > "${OUT}/suites_spill_diff.tsv" 2>&1 || true
    stop_server
}

## ---------------------------------------------------------------------------------------------
main() {
    [ -s "${ACC}" ] || printf 'arm\tcfg\tshape\tbuild_rows\treported_after_first_block\tpeak_reported_bytes\tpeak_memory_bytes\ttag\n' > "${ACC}"
    [ -s "${SPILL}" ] || printf 'arm\tcfg\tbuild_rows\tspilled\tpeak_memory_bytes\tsec\tjoin_build_rows\ttag\n' > "${SPILL}"

    hr; echo "PASS 1: what getTotalByteCount reports, in bytes (bisection on max_bytes_in_join)"; hr
    if want_arm baseline; then run_accounting_for_arm baseline; fi
    if want_arm uhj;      then run_accounting_for_arm uhj;      fi

    if [ "${SKIP_SPILL:-0}" != 1 ]; then
        hr; echo "PASS 2: spill point at max_bytes_before_external_join=${SPILL_CAP}"; hr
        if want_arm baseline; then run_spill_for_arm baseline; fi
        if want_arm uhj;      then run_spill_for_arm uhj;      fi
    fi

    if [ "${SKIP_SUITES:-0}" != 1 ]; then
        hr; echo "PASS 3: four suites, unbounded vs capped at ${SUITE_CAP}"; hr
        if want_arm baseline; then run_suites_for_arm baseline; fi
        if want_arm uhj;      then run_suites_for_arm uhj;      fi
        summarize_suites
    fi

    hr; echo "SUMMARY m_${ID} - bucket_bytes accounting"; hr
    echo "-- reported bytes vs real peak memory (build-only queries) --"
    printf '%-8s %-9s %10s %16s %16s %16s\n' cfg shape rows after_1st_blk peak_reported peak_memory
    awk -F'\t' 'NR > 1 { printf "%-8s %-9s %10s %16s %16s %16s\n", $2, $3, $4, $5, $6, $7 }' "${ACC}" \
        | sort -k2,2 -k1,1
    echo
    echo "   accounting gap, UHJ against the baseline configuration with the same map layout"
    echo "   (umt1 vs bh: both one flat table; umt16 vs bph: both 256-bucket):"
    awk -F'\t' '
        NR > 1 { v[$2 "|" $3] = $6 }
        END {
            split("umt1:bh umt16:bph", pairs, " ")
            for (i in pairs) {
                split(pairs[i], p, ":")
                for (k in v) {
                    split(k, a, "|")
                    if (a[1] != p[1]) continue
                    ref = p[2] "|" a[2]
                    if (ref in v)
                        printf "     %-9s %-6s vs %-6s %14d bytes\n", a[2], p[1], p[2], v[ref] - v[k]
                }
            }
        }' "${ACC}" | sort -k1,1 -k2,2
    if [ "$(wc -l < "${SPILL}")" -gt 1 ]; then
        echo
        echo "-- spill point (cap=${SPILL_CAP}; SpillingHashJoin switches at half the cap) --"
        printf '%-8s %10s %8s %14s %9s\n' cfg build_rows spilled peak_memory sec
        awk -F'\t' 'NR > 1 { printf "%-8s %10s %8s %14s %9s\n", $2, $3, $4, $5, $6 }' "${SPILL}" \
            | sort -k2,2n -k1,1
        echo
        echo "   first build size that spills, per configuration:"
        awk -F'\t' 'NR > 1 && $4 > 0 { if (!(($2) in m) || $3 + 0 < m[$2]) m[$2] = $3 + 0 }
                    END { for (c in m) printf "     %-8s %d rows\n", c, m[c] }' "${SPILL}" | sort
    fi
    if [ -s "${OUT}/suites.tsv" ]; then
        echo
        echo "-- four suites, bound n=unbounded / c=capped at ${SUITE_CAP}, arm b=baseline / u=uhj --"
        column -t -s $'\t' "${OUT}/suites.tsv" || cat "${OUT}/suites.tsv"
        if [ -s "${OUT}/suites_spill_diff.tsv" ] && [ "$(wc -l < "${OUT}/suites_spill_diff.tsv")" -gt 1 ]; then
            echo
            echo "   queries that spill on one arm only under the same cap:"
            column -t -s $'\t' "${OUT}/suites_spill_diff.tsv" || cat "${OUT}/suites_spill_diff.tsv"
        fi
    fi
    hr
    echo "artifacts: ${OUT}"
    echo "M_${ID}_DONE"
}

main "$@"
