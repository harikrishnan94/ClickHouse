#!/usr/bin/env bash
# D16 -- UHJ's LowCardinality emplaceKey decodes the dictionary index twice per build row, and,
# when the dictionary has no saved hash, builds the key holder twice and hashes it twice
# (UnifiedHashJoin/KeyGetter.h:119-137). The baseline passes saved_hash[row] straight to emplace
# and decodes once (HashJoin/KeyGetter.h:136-143).
#
# The second hash is dead code. tryGetLowCardinalityMethod admits only String and FixedString
# nested types, ReverseIndex::use_saved_hash is !is_numeric_column, and tryGetSavedHash
# materialises the hash array on first call rather than returning null (Columns/ReverseIndex.h:349).
# So saved_hash is never null for a dictionary the join can reach, and what remains live is one
# extra getIndexAt -- a switch on size_of_index_type plus an indexed load -- which the compiler may
# CSE away entirely, since both calls read the same const column element with no intervening store.
#
# This script therefore measures a quantity that may legitimately be zero, and is built to tell
# zero apart from unmeasured. The discriminator is key width: an extra string hash scales with it,
# an extra index decode does not. Same dictionary, same rows, two widths.
#
# max_threads = 1 throughout the primary matrix. It is the only setting at which unified_hash and
# baseline hash build the SAME single-level low_cardinality_key_string map with one slot, one
# getter and no scatter, so the build inner loop is the only thing that differs. At mt > 1 the arms
# diverge on the map itself and D16 disappears into D8.
#
# See SPEC_MAPS.md section 5.
#
# Usage: ./m_D16.sh           (REPS, WANT_ARM overridable in the environment)
set -euo pipefail

# Deterministic instruction counts matter more than clock here, and the effect is per row, so
# spend the repetitions on stability rather than on breadth. Set before sourcing, because the
# harness applies its own default of 5 to an unset REPS.
REPS="${REPS:-7}"

# shellcheck source=_maps_common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_maps_common.sh"

maps_take_lock
maps_init D16
trap 'stop_server' EXIT

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

build_q() { printf 'SELECT count() FROM %s AS l INNER JOIN %s AS r ON l.k = r.k' "$2" "$1"; }

cases() {
    # --- the width discriminator. Same 1000-entry dictionary, same 50 M rows, 16-byte versus
    # --- 48-byte keys. If the unified_hash minus hash gap per build row is the same at both
    # --- widths, the second hash is not being paid and the source reading above is confirmed.
    do_case lc_w16  1 "$(build_q lc_build_w16_d1k lc_nomatch)"
    do_case lc_w48  1 "$(build_q lc_build_w48_d1k lc_nomatch)"
    # --- control: the same 48-byte strings as a plain String column, so neither arm uses a
    # --- LowCardinality key getter. Any gap here is the rest of the fork, and the reportable
    # --- number for D16 is the difference of differences against it.
    do_case str_w48 1 "$(build_q str_build_w48_d1k str_nomatch)"

    # --- dictionary size selects the getIndexAt branch: UInt8 below 256 entries, UInt16 below
    # --- 65536, UInt32 beyond. A real per-row cost should be roughly constant across these; one
    # --- that moves a lot is measuring cache behaviour of saved_hash[row], which both arms pay.
    # --- These are 20 M rows against the 50 M above, which doubles as the linearity check:
    # --- the per-row gap must not depend on the row count.
    do_case lc_sweep_d1k   1 "$(build_q lc_sweep_d1k   lc_nomatch)"
    do_case lc_sweep_d100k 1 "$(build_q lc_sweep_d100k lc_nomatch)"
    do_case lc_sweep_d1m   1 "$(build_q lc_sweep_d1m   lc_nomatch)"

    # --- context only, and NOT D16: at sixteen threads unified_hash switches to
    # --- two_level_low_cardinality_key_string while baseline hash stays single-level, so this is
    # --- D8 territory. Included so the mt=1 numbers can be put in proportion.
    do_case lc_w48_mt16 16 "$(build_q lc_build_w48_d1k lc_nomatch)"
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

# The reportable figure is instructions per build row, not seconds: an extra getIndexAt is an
# instruction-count question and the instruction count is deterministic where the clock is not.
if [ -s "${QLOG_TSV}" ]; then
    echo
    echo "=== D16: instructions per build row (the figure to report) ==="
    awk -F'\t' 'NR==1 { for (i=1;i<=NF;i++) c[$i]=i; next }
        { printf "%-16s %-13s mt=%-3s  %14s  %14s\n", $c["case_name"], $c["algo"], $c["max_threads"],
                 $c["instr_per_build_row"], $c["build_rows"] }' "${QLOG_TSV}" \
        | (printf '%-16s %-13s %-6s  %14s  %14s\n' case algo mt instr_per_row build_rows; sort)
fi

cat <<'EOF'

How to read this
  * maptypes.txt must show datatype: low_cardinality_key_string on BOTH unified_hash and hash for
    every mt=1 case. That agreement is what makes the isolation possible; if the arms disagree on
    the map, this script measured D8, not D16.
  * The discriminator: compare (unified_hash - hash) instructions per build row at lc_w16 against
    the same quantity at lc_w48. Equal means the extra key holder and extra hash are unreachable,
    as the source says, and D16's live cost is at most one extra index decode. Growing with width
    means saved_hash is null somewhere and the second hash is live -- which would contradict
    Columns/ReverseIndex.h:324 and should be chased down before being reported.
  * Subtract str_w48. The unified_hash minus hash gap on a LowCardinality build at one thread
    contains every other single-threaded build-path divergence; the difference of differences is
    the closest this can get to D16 alone, and even that keeps whatever else is specific to the
    LowCardinality getter.
  * Expect a number below one percent of build time, possibly exactly zero. Report zero as zero.
    A per-row cost that cannot be resolved above run-to-run variance should be stated as an upper
    bound, with the variance quoted, rather than as a point estimate.
  * lc_w48_mt16 is D8, labelled here only to keep the mt=1 figures in proportion.
EOF

stop_server
trap - EXIT
echo
echo "M_D16_DONE"
