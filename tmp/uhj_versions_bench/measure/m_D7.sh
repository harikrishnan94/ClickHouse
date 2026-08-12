#!/usr/bin/env bash
# D7 - offset computation moved from the lazy `offsetInternal` (a `std::call_once` plus a re-hash of
#      the stored key to recover its bucket) to `offsetInternalUnsafe` / `offsetInternalAtBucket`
#      after an explicit `computeBucketPrefix` barrier (`UnifiedHashJoin/HashJoin.cpp:2070-2082`).
#
# Where this is actually live, from reading both trees - this is narrower than the inventory implies
# and it decides the whole design:
#
#   * Probe path, generic key getters (key64, keys256, key_string, ...): NOT live. Those getters
#     reach the offset through the shared `ColumnsHashingImpl::findKeyImpl`, which calls
#     `data.offsetInternal(it)` on both arms. On the UHJ binary that is the branch's
#     `TwoLevelHashTable::offsetInternal`, which for bucketCount() > 1 still re-hashes
#     (`getBucketFromHash(bucketRoutingHash(ptr->getKey(), ptr->getHash(*this)))`) and still goes
#     through `BucketPrefixSums::offset`, i.e. through the `std::call_once`. Only
#     `LowCardinalityKeyGetterForJoin::findKey` calls `offsetInternalUnsafe` directly
#     (`UnifiedHashJoin/KeyGetter.h:167`).
#   * Probe path, LowCardinality getter: live in principle, unobservable in practice. Baseline
#     forbids the dictionary-aware map whenever `use_two_level_maps` (`HashJoin/HashJoin.cpp:202`),
#     so at max_threads > 1 baseline does not use that getter at all (that asymmetry is D8), and at
#     max_threads = 1 both arms have a single-bucket map, where `offsetInternalUnsafe` and
#     `offsetInternal` are the same pointer subtraction. There is no configuration in which the two
#     arms run the same LowCardinality getter over a multi-bucket map, so this component is not
#     measurable with the two binaries and is not measured here.
#   * Non-joined scan: live, and this is what the script measures. Baseline scans with
#     `map.offsetInternal(it.getPtr())` (`HashJoin/HashJoin.cpp:1431, 1450`), which re-derives the
#     bucket by re-hashing every live cell's key; UHJ scans with
#     `map.offsetInternalAtBucket(it.getPtr(), it.getBucket())`
#     (`UnifiedHashJoin/HashJoin.cpp:1489`), which takes the bucket from the iterator and reads the
#     prefix sums that `freezeMapsForProbing` already computed. One key re-hash plus one
#     `call_once` acquire load per live map cell, saved.
#   * `computeBucketPrefix` barrier: 256 iterations, run twice per join (`onBuildPhaseFinish` and
#     the end of `runPostBuildPhase`). 512 loop iterations against a build of millions of rows; not
#     measurable, not measured, stated in SPEC_PROBE.md instead.
#
# Worst case therefore: a RIGHT/FULL join that (a) needs the scan, (b) has as many live cells as
# possible, and (c) has the most expensive key hash, since the saving is one re-hash per cell.
# RIGHT ANTI with every right row matched gives a scan that visits every cell and emits nothing, so
# the measurement is the scan and only the scan; `keys256` (four UInt64 key columns hashed with
# UInt256HashCRC32) is the dearest re-hash; `key64` (HashCRC32 over 8 bytes) is the cheap contrast.
# The signature of D7 is that the arm gap grows from key64 to keys256; a gap that does not move with
# the hash cost is something else (the scan loop, the collector, D4).
#
# Results: /mnt/data/uhj_versions_bench/measure/D7/
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_probe_common.sh"

ID=D7
OUT="$(m_out_dir "${ID}")"
REPEATS="${REPEATS:-3}"
MT_LIST="${MT_LIST:-1 8 16}"

RES="${OUT}/results.tsv"
MAPS="${OUT}/maptypes.tsv"

m_take_lock

suite_census "${OUT}/suite_census.tsv"
suite_census_summary "${OUT}/suite_census.tsv" | tee "${OUT}/suite_census_summary.txt"
echo "# D7 exposure: the non-joined scan runs only for RIGHT/FULL joins - see those columns."

N1="${PB_K256_BUILD}"       # 1:1 probe: every build key matched exactly once, zero anti output
HALF=$(( PB_K256_BUILD / 2 ))

## -------------------------------------------------------------------------------------------
## Shapes. <key> in {u64, k256}; <cov> is how much of the build side the probe covers.
##   full - every right row matched: scan visits every cell, emits nothing
##   half - half matched: scan visits every cell, emits half the rows
##   none - nothing matched: scan visits every cell, emits everything (upper bound)
## -------------------------------------------------------------------------------------------
sql_cell() {   # sql_cell <key> <cov>
    local key="$1" cov="$2" l r on lim
    if [ "${key}" = k256 ]; then
        r="pb_k256_build"; l="pb_k256_probe"; on="l.a = r.a AND l.b = r.b AND l.c = r.c AND l.d = r.d"
    else
        r="pb_u64_build"; l="pb_u64_probe"; on="l.a = r.a"
    fi
    case "${cov}" in
        full) lim="i < ${N1}" ;;
        half) lim="i < ${HALF}" ;;
        none) lim="i >= ${PB_K256_PROBE}" ;;   # an empty probe side: no flag is ever set
        *) echo "unknown coverage '${cov}'" >&2; return 1 ;;
    esac
    echo "SELECT count() FROM (SELECT * FROM ${l} WHERE ${lim}) AS l RIGHT ANTI JOIN ${r} AS r ON ${on}"
}

run_cell() {   # run_cell <algo> <key> <cov> <max_threads>
    local algo="$1" key="$2" cov="$3" mt="$4"
    local tag="pb${ID}_$(algo_code "${algo}")_${key}${cov}_mt${mt}"
    local sql best all nj_us nj_streams nj_rows build_rows instr per_cell
    sql="$(sql_cell "${key}" "${cov}")"
    q_warm "${SYNTH_DB}" "${sql}" "--join_algorithm=${algo}" "--max_threads=${mt}"
    read -r best all <<<"$(q_best "${REPEATS}" "${tag}" "${SYNTH_DB}" "${sql}" \
        "--join_algorithm=${algo}" "--max_threads=${mt}")"
    nj_us="$(pp_metric "${tag}" NonJoinedBlocksTransform 'toUInt64(sum(elapsed_us) / greatest(count(distinct query_id), 1))')"
    nj_streams="$(pp_metric "${tag}" NonJoinedBlocksTransform 'toUInt64(count() / greatest(count(distinct query_id), 1))')"
    nj_rows="$(pp_metric "${tag}" NonJoinedBlocksTransform 'toUInt64(sum(output_rows) / greatest(count(distinct query_id), 1))')"
    build_rows="$(qlog_metric "${tag}" "toUInt64(median(ProfileEvents['JoinBuildTableRowCount']))")"
    instr="$(qlog_metric "${tag}" "toUInt64(median(ProfileEvents['PerfInstructions']))")"
    # Cells, not rows: with unique build keys the two coincide, which is why the fixtures are unique.
    per_cell="$(awk -v a="${nj_us}" -v b="${build_rows}" 'BEGIN { printf "%.6f", (b > 0) ? a * 1000 / b : 0 }')"
    tsv_prune "${RES}" "${tag}"
    tsv "${RES}" "${tag}" "${CUR_ARM}" "${algo}" "${key}" "${cov}" "${mt}" "${best}" "${nj_us}" \
        "${nj_streams}" "${nj_rows}" "${build_rows}" "${per_cell}" "${instr}" "${all}"
    printf '  %-38s best=%-8s nonjoined_us=%-9s ns/cell=%-10s emitted=%s\n' \
        "${tag}" "${best}" "${nj_us}" "${per_cell}" "${nj_rows}"
}

## -------------------------------------------------------------------------------------------
## Arms
## -------------------------------------------------------------------------------------------
tsv_head "${RES}" tag arm algo key coverage max_threads best_sec nonjoined_us nonjoined_streams \
    nonjoined_rows build_rows nonjoined_ns_per_cell instructions all_times
tsv_head "${MAPS}" tag arm algo key max_threads map_type

T_START="$(now_epoch)"

for arm in baseline uhj; do
    want_arm "${arm}" || continue
    start_server "${arm}"
    ensure_probe_synth k256

    MEASURE_SETTINGS=()
    perfev_available && MEASURE_SETTINGS=("${PERFEV_SETTINGS[@]}")

    # `hash` cannot be compared on the non-joined phase: with a single stream the pipeline emits
    # non-joined rows from inside JoiningTransform, so there is no NonJoinedBlocksTransform to read.
    # It is measured anyway, at max_threads=1, as the whole-query reference point.
    if [ "${arm}" = baseline ]; then algos=(parallel_hash hash); else algos=(unified_hash); fi

    for algo in "${algos[@]}"; do
        if [ "${algo}" = hash ]; then mts="1"; else mts="${MT_LIST}"; fi

        for mt in ${mts}; do
            for key in u64 k256; do
                mtype="$(q_maptype "${SYNTH_DB}" "$(sql_cell "${key}" full)" \
                    "--join_algorithm=${algo}" "--max_threads=${mt}")"
                tag="pb${ID}map_$(algo_code "${algo}")_${key}_mt${mt}"
                tsv_prune "${MAPS}" "${tag}"
                tsv "${MAPS}" "${tag}" "${arm}" "${algo}" "${key}" "${mt}" "${mtype}"
                echo "# map: arm=${arm} algo=${algo} key=${key} mt=${mt} -> ${mtype}"
            done
        done

        hr; echo "# ${ID} non-joined scan: arm=${arm} algo=${algo}"
        for mt in ${mts}; do
            for key in u64 k256; do
                for cov in full half none; do
                    run_cell "${algo}" "${key}" "${cov}" "${mt}"
                done
            done
        done
    done

    qlog_agg  "pb${ID}" "${T_START}" "${OUT}/querylog_${arm}.tsv"
    pplog_agg "pb${ID}" "${T_START}" "${OUT}/processors_${arm}.tsv"
done

## -------------------------------------------------------------------------------------------
## Summary
## -------------------------------------------------------------------------------------------
{
    hr
    echo "D7 summary - non-joined scan, nanoseconds per live map cell (${N1} cells)"
    hr
    awk -F'\t' '
        NR == 1 { next }
        { v[$3 SUBSEP $4 SUBSEP $6 SUBSEP $5] = $12; seen[$4 SUBSEP $6 SUBSEP $5] = 1 }
        END {
            printf "%-5s %4s %-5s %14s %14s %10s %10s\n",
                   "key", "mt", "cov", "ph_ns/cell", "uh_ns/cell", "delta", "ratio"
            for (k in seen) {
                split(k, p, SUBSEP); key = p[1]; mt = p[2]; cov = p[3]
                b = v["parallel_hash" SUBSEP key SUBSEP mt SUBSEP cov]
                u = v["unified_hash" SUBSEP key SUBSEP mt SUBSEP cov]
                if (b == "" || u == "") continue
                printf "%-5s %4s %-5s %14s %14s %10.4f %10.3f\n", key, mt, cov, b, u, b - u,
                       (u > 0) ? b / u : 0
            }
            print ""
            print "D7 predicts delta > 0 (UHJ cheaper) and delta(k256) > delta(u64), because what is"
            print "saved is one re-hash of the stored key per live cell plus one call_once acquire."
            print "A delta that does not grow with the key hash cost is not D7."
            print "At max_threads=1 the two arms do not have the same map (parallel_hash keeps a"
            print "256-bucket map with slots=1 while UHJ has a flat one), so that row mixes D7 with"
            print "divergence B and must be quoted separately."
        }' "${RES}"
    echo
    echo "Full table:"
    column -t -s $'\t' "${RES}" 2>/dev/null || cat "${RES}"
    echo
    echo "Map layouts actually measured:"
    column -t -s $'\t' "${MAPS}" 2>/dev/null || cat "${MAPS}"
    echo
    echo "Real-world exposure:"
    cat "${OUT}/suite_census_summary.txt"
} | tee "${OUT}/summary.txt"

done_sentinel "${ID}" "${OUT}"
