#!/usr/bin/env bash
# D8 -- UHJ permits the dictionary-aware LowCardinality map in a parallel build; the baseline
# forbids it whenever use_two_level_maps (HashJoin/HashJoin.cpp:202 has `&& !use_two_level_maps`,
# absent at UnifiedHashJoin/HashJoin.cpp:346) and materialises the key column instead.
#
# So the three arms build three different things for one query, which is the point:
#   hash          low_cardinality_key_string             (serial, dictionary-aware)
#   parallel_hash two_level_key_string over materialised String
#   unified_hash  two_level_low_cardinality_key_string   (parallel, dictionary-aware)
#
# See SPEC_MAPS.md section 2. The sign of the result turns on one ratio -- dictionary size versus
# rows per block -- so the dictionary sweep is the measurement, not a robustness check. Below a
# block the per-block visit_cache turns thousands of lookups into a handful; above it the cache is
# allocated, zeroed and never reused.
#
# Usage: ./m_D8.sh            (REPS, THREADS, WANT_ARM overridable in the environment)
set -euo pipefail

# shellcheck source=_maps_common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_maps_common.sh"

maps_take_lock
maps_init D8
trap 'stop_server' EXIT

THREADS="${THREADS:-1 4 16}"

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

# probe_q <probe> <dim>: probe-heavy, every probe row matches exactly one build row, so the
# result size is identical across the dictionary sweep and only the lookup path changes.
probe_q() { printf 'SELECT count(), sum(r.v) FROM %s AS l INNER JOIN %s AS r ON l.k = r.k' "$1" "$2"; }

# build_q <build> <probe>: build-heavy. The four-row probe matches nothing, so the build side is
# materialised in full, the probe costs nothing and the join emits no rows.
build_q() { printf 'SELECT count() FROM %s AS l INNER JOIN %s AS r ON l.k = r.k' "$2" "$1"; }

cases() {
    local mt
    for mt in ${THREADS}; do
        # --- probe-heavy: 50 M probe rows, dictionary from 16 to 1 M ------------------------
        # 16 entries is 1/4000 of a block, so visit_cache collapses a whole block's lookups into
        # sixteen; 1 M is fifteen blocks' worth, so nearly every cache slot is touched once.
        do_case lc_d16   "${mt}" "$(probe_q lc_probe_d16   lc_dim_d16)"
        do_case lc_d1k   "${mt}" "$(probe_q lc_probe_d1k   lc_dim_d1k)"
        do_case lc_d100k "${mt}" "$(probe_q lc_probe_d100k lc_dim_d100k)"
        do_case lc_d1m   "${mt}" "$(probe_q lc_probe_d1m   lc_dim_d1m)"
        # The materialised twin of lc_d1k: same bytes, plain String columns. This is what
        # parallel_hash's LowCardinality path effectively degenerates to, priced within one arm
        # so the difference is not attributed to the arm.
        do_case str_d1k  "${mt}" "$(probe_q str_probe_d1k  str_dim_d1k)"

        # --- build-heavy: 20 M build rows, dictionary the only axis -------------------------
        do_case lcb_d1k   "${mt}" "$(build_q lc_sweep_d1k   lc_nomatch)"
        do_case lcb_d10k  "${mt}" "$(build_q lc_sweep_d10k  lc_nomatch)"
        do_case lcb_d100k "${mt}" "$(build_q lc_sweep_d100k lc_nomatch)"
        do_case lcb_d1m   "${mt}" "$(build_q lc_sweep_d1m   lc_nomatch)"
        do_case strb_d1k  "${mt}" "$(build_q str_sweep_d1k  str_nomatch)"
        do_case strb_d1m  "${mt}" "$(build_q str_sweep_d1m  str_nomatch)"
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
arm hash
arm parallel_hash
flush_logs

start_server uhj
arm unified_hash

collect_stages
collect_qlog
summarize

cat <<'EOF'

How to read this
  * maptypes.txt must show three different maps for the lc_* cases at max_threads > 1:
    low_cardinality_key_string on hash, two_level_key_string on parallel_hash,
    two_level_low_cardinality_key_string plus a [dictionary-aware] tag on unified_hash. If
    parallel_hash reports a low_cardinality_* map, the baseline binary is wrong.
  * max_threads = 1 is the zero point: there unified_hash and hash build the same
    low_cardinality_key_string map, so whatever gap remains at mt=1 is not D8 and has to be
    subtracted from the parallel rows.
  * The finding is the crossover: the dictionary size at which unified_hash stops beating
    parallel_hash on the probe-heavy rows. Predicted near dictionary = block size (65409).
  * str_d1k separates the two things unified_hash avoids -- the string materialisation and the
    repeated lookups. The lc_d1k / str_d1k ratio within one arm is the dedup benefit alone.
  * The build-heavy lcb_* rows contain D13's per-slot cache construction, which is not shared
    for the LowCardinality getter. Read them together with m_D13.sh; they cannot be separated
    on this axis.
EOF

stop_server
trap - EXIT
echo
echo "M_D8_DONE"
