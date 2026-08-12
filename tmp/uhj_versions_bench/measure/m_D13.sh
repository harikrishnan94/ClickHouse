#!/usr/bin/env bash
# D13 -- BlockKeyGetter shares one key getter across slots when the getter declares
# reads_whole_block_at_construction (UnifiedHashJoin/HashJoinMethods.h:102-131). No baseline
# counterpart: in ConcurrentHashJoin every shard is a separate HashJoin building its own getter
# over its own scattered piece of the block.
#
# The inventory attributes the flag to the LowCardinality getter. It is wrong. The only
# declaration in the tree is on HashMethodKeysFixed (Common/ColumnsHashing/HashMethod.h:411), the
# composite fixed-width key getter, whose constructor packs the entire block into prepared_keys.
# LowCardinalityKeyGetterForJoin does not declare it and does not inherit it, so its
# dictionary-sized visit_cache / mapped_cache / offset_cache are rebuilt per slot per block.
# D13 therefore helps composite-key joins and silently costs LowCardinality ones.
#
# There is no A/B: shareKeyGetterAcrossBuckets is constexpr. The measurement is a counterfactual
# by contrast -- compare key shapes where sharing is active against shapes where it is not, and
# read the SLOPE in max_threads, which is exactly what sharing removes. A per-slot constructor
# cost C appears as C x slots per block; a shared one appears as C per block whatever the slots.
#
# See SPEC_MAPS.md section 4. The deliverable is d(build_us)/d(slots) per key shape, not a ratio.
#
# Usage: ./m_D13.sh           (REPS, THREADS, WANT_ARM overridable in the environment)
set -euo pipefail

# shellcheck source=_maps_common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_maps_common.sh"

maps_take_lock
maps_init D13
trap 'stop_server' EXIT

THREADS="${THREADS:-1 2 4 8 16}"

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

# Build-heavy throughout: a four-row probe that matches nothing, so build cost is the query and
# nothing on the output side varies between points.
build1_q() { printf 'SELECT count() FROM %s AS l INNER JOIN %s AS r ON l.k = r.k' "$2" "$1"; }
build2_q() { printf 'SELECT count() FROM %s AS l INNER JOIN %s AS r ON l.a = r.a AND l.b = r.b' "$2" "$1"; }
build4_q() {
    printf 'SELECT count() FROM %s AS l INNER JOIN %s AS r ON l.a = r.a AND l.b = r.b AND l.c = r.c AND l.d = r.d' "$2" "$1"
}

cases() {
    local mt
    for mt in ${THREADS}; do
        # --- sharing NOT active: LowCardinality. Rows are constant at 20 M and only the
        # --- dictionary grows, so any build-time growth is per-slot constructor cost and
        # --- nothing else. Each constructor allocates and zeroes visit_cache (1 B/entry),
        # --- mapped_cache (8 B/entry) and offset_cache (8 B/entry) sized by the whole
        # --- dictionary, not by the rows in that slot.
        do_case lc_d1k   "${mt}" "$(build1_q lc_sweep_d1k   lc_nomatch)"
        do_case lc_d10k  "${mt}" "$(build1_q lc_sweep_d10k  lc_nomatch)"
        do_case lc_d100k "${mt}" "$(build1_q lc_sweep_d100k lc_nomatch)"
        do_case lc_d1m   "${mt}" "$(build1_q lc_sweep_d1m   lc_nomatch)"

        # --- the control that makes the slope readable: identical data as plain String, whose
        # --- getter constructor is O(1). Its build time must not move with dictionary size, and
        # --- whatever slope it does show is the floor to subtract from the rows above.
        do_case str_d1k "${mt}" "$(build1_q str_sweep_d1k str_nomatch)"
        do_case str_d1m "${mt}" "$(build1_q str_sweep_d1m str_nomatch)"

        # --- sharing active: two UInt64s are sixteen bytes per row, above the eight-byte
        # --- dense_keys threshold in scatterBlockBySlot, so every slot goes through the shared
        # --- getter and the block is packed into UInt128 once instead of `slots` times.
        do_case keys128 "${mt}" "$(build2_q d13_build_keys128 d13_nomatch_keys128)"

        # --- sharing bypassed: two UInt32s are exactly eight bytes, so scatterBlockBySlot emits
        # --- dense_keys and insertFromBlockImplTypeCase builds a private getter over the
        # --- scattered columns, never consulting the shared one. Total pack work is still one
        # --- pass over the rows, so this should also be flat -- a finding about
        # --- scatterBlockBySlot rather than noise.
        do_case keys64  "${mt}" "$(build2_q d13_build_keys64 d13_nomatch_keys64)"

        # --- shared, but sizeof(UInt256) > 16 so usePreparedKeys is false and the constructor is
        # --- cheap anyway: the null case, which should match keys128's slope.
        do_case keys256 "${mt}" "$(build4_q d13_build_keys256 d13_nomatch_keys256)"
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
# The baseline arms are context, not the subject: parallel_hash shows what the merge base pays
# instead (a real scatter of the key columns, then one cheap getter per shard over 1/N of the
# rows) and hash gives the single-threaded floor.
arm hash
arm parallel_hash
flush_logs

start_server uhj
arm unified_hash

collect_stages
collect_qlog
summarize

# The slope is the deliverable, so compute it here rather than leaving it to the reader. Simple
# least squares of build_us against slot count, per case and arm, over the max_threads points.
# slots = min(bit_ceil(max_threads), 256), which for this sweep is max_threads itself.
if [ -s "${STAGE_TSV}" ]; then
    echo
    echo "=== D13: build_us regressed on slot count (median over reps) ==="
    awk -F'\t' '
        function med(s,   n,a,i,j,t) { n=split(s,a," "); if(!n) return 0;
            for(i=1;i<n;i++) for(j=i+1;j<=n;j++) if(a[j]+0<a[i]+0){t=a[i];a[i]=a[j];a[j]=t}
            return a[int((n+1)/2)]+0 }
        NR>1 { v[$1"\t"$2"\t"$3] = v[$1"\t"$2"\t"$3]" "$5 }
        END {
            for (k in v) { split(k,a,"\t"); key=a[1]"\t"a[2]
                x[key]=x[key]" "a[3]; y[key]=y[key]" "med(v[k]) }
            printf "%-12s %-13s %12s %12s %10s\n", "case", "algo", "intercept_us", "slope_us", "points"
            for (key in x) {
                n=split(x[key],xs," "); split(y[key],ys," ")
                if (n < 2) continue
                sx=sy=sxx=sxy=0
                for (i=1;i<=n;i++) { sx+=xs[i]; sy+=ys[i]; sxx+=xs[i]*xs[i]; sxy+=xs[i]*ys[i] }
                den = n*sxx - sx*sx
                if (den == 0) continue
                slope = (n*sxy - sx*sy) / den
                icept = (sy - slope*sx) / n
                split(key,a,"\t")
                printf "%-12s %-13s %12.0f %12.1f %10d\n", a[1], a[2], icept, slope, n
            }
        }' "${STAGE_TSV}" | (read -r h; echo "${h}"; sort)
fi

cat <<'EOF'

How to read this
  * The slope table above is the finding. Expected: a clearly positive slope for the lc_* rows on
    unified_hash that grows with dictionary size, and a slope near zero for str_*, keys128,
    keys64 and keys256. A flat lc_d1m contradicts the source reading in SPEC_MAPS.md section 4
    and means the caches are being shared after all.
  * The slope is contaminated by everything else that scales with slot count -- arena growth,
    bucket_bytes accounting (D2), lock traffic (D9). str_d1k is the subtraction term: it shares
    nothing, has an O(1) constructor, and its slope is the floor.
  * keys64 looking like keys128 is not a null result. It means scatterBlockBySlot's dense_keys
    path made the sharing decision moot for eight-byte composite keys, which is worth stating.
  * D13 buys unified_hash parity with the baseline on composite keys, not an advantage: the
    baseline's per-shard getters also pack each row exactly once in total. The magnitude worth
    quoting is the counterfactual -- what UHJ would pay per block without sharing, which is the
    lc_* slope scaled to the keys128 constructor cost.
EOF

stop_server
trap - EXIT
echo
echo "M_D13_DONE"
