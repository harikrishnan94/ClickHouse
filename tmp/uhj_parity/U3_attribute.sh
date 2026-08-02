#!/usr/bin/env bash
# U3: attribute the normalized residual diff to inventory categories at HUNK granularity.
#
# Line granularity undercounts: a hunk's identifying token (e.g. `bucket`) sits on one line while
# the hunk spans a whole multi-line signature or body. So classify each `@@` hunk as a unit, and
# report the hunks that carry NO marker at all -- that residue is where divergence not explained by
# compulsory TwoLevel would be hiding, and it is the only part that needs human eyes.
set -u
cd /mnt/ch/ClickHouse

IN=tmp/uhj_parity/U3_normdiff.txt
OUT=tmp/uhj_parity/U3_unattributed_hunks.txt

TWOLEVEL='bucket|Bucket|BUCKETS_PER_THREAD|BITS_FOR_BUCKET|num_buckets|impls|prober|Prober|computeBucketPrefix|freezeMapsForProbing|recomputeBucketBytes|JoinHashMap|JoinFixedHashMap|PartitionedFixedHashMap|BucketPartitioned|TwoLevel|two_level|scatterBy|insertIntoBuckets|BuildResult|new_keys|needs_offset|use_offset|getOffset|offsetInternal|offset_cache|routingHashForRow|CacheLine|max_threads|poolForBucket|pools|FixedRangeStorage|getBucketFromHash|bucketRoutingHash|prefetchByHash|cells|chooseMethod|mergeJoinMethods|create.data->type'

PARBUILD='atomic|mutex|lock|Lock|Unlocked|memory_order|shrink_blocks|all_values_unique|is_inserted|keys_to_join|rows_to_join|allocated_size|ScatteredBlock|joinScatteredBlock|getUsedFlags|setUsedFlags|hasNonJoinedRows|updateNonJoinedRowsStatus|allOffsetFlagsSet|supportParallelJoin|stored_columns_it|setTotals|getTotals|doDebugAsserts|getTotalRowCount|getTotalByteCount'

FORKMECH='UNIFIED_APPLY_FOR_JOIN_VARIANTS|getName|UnifiedHashJoin|#include|<algorithm>|<bit>|<mutex>|JoinSource'

: > "$OUT"
awk -v tl="$TWOLEVEL" -v pb="$PARBUILD" -v fm="$FORKMECH" -v out="$OUT" '
function flush(  cat) {
    if (nh == 0) return
    cat = "UNATTRIBUTED"
    if      (body ~ tl) cat = "TWOLEVEL"
    else if (body ~ pb) cat = "PARALLEL_BUILD"
    else if (body ~ fm) cat = "FORK_MECHANICAL"
    hunks[cat]++; lines[cat] += nchanged
    if (cat == "UNATTRIBUTED") {
        printf "=== %s  %s ===\n%s\n", file, header, body >> out
    }
    nh = 0; nchanged = 0; body = ""
}
/^=== DIFF /   { flush(); file = $3; next }
/^@@/          { flush(); header = $0; nh = 1; next }
nh == 1        { body = body $0 "\n"; if ($0 ~ /^[+-][^+-]/) nchanged++ }
END {
    flush()
    split("TWOLEVEL PARALLEL_BUILD FORK_MECHANICAL UNATTRIBUTED", order, " ")
    for (i = 1; i <= 4; i++) {
        k = order[i]
        printf "%-16s hunks=%-4d changed_lines=%d\n", k, hunks[k], lines[k]
    }
}' "$IN"

echo "unattributed hunks written to: $OUT"
