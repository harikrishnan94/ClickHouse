#!/usr/bin/env bash
# Robustness check on U3_attribute.sh: the verifier observed that its markers are broad
# (`lock`, `offset`, `cells`, `pools`, `#include`, `max_threads` ...), so `UNATTRIBUTED=0` could be
# an artifact of the regex rather than a property of the code. This re-runs the same hunk-level
# attribution with only NARROW, unambiguous TwoLevel markers, and dumps whatever falls out for
# manual review. A small, reviewable residue that is still genuinely TwoLevel on inspection means
# the original conclusion does not depend on the broad markers.
set -u
cd /mnt/ch/ClickHouse
IN=tmp/uhj_parity/U3_normdiff.txt
OUT=tmp/uhj_parity/U3_strict_residue.txt

# Narrow markers only: each names a construct that exists solely because the map is partitioned.
STRICT='bucket|Bucket|BUCKETS_PER_THREAD|BITS_FOR_BUCKET|impls|scatterBy|insertIntoBuckets|JoinHashMap|JoinFixedHashMap|PartitionedFixedHashMap|BucketPartitioned|TwoLevel|two_level|prober|Prober|freezeMapsForProbing|needs_offset|use_offset|BuildResult|poolForBucket'

: > "$OUT"
awk -v st="$STRICT" -v out="$OUT" '
function flush(  cat) {
    if (nh == 0) return
    cat = (body ~ st) ? "STRICT_TWOLEVEL" : "RESIDUE"
    hunks[cat]++; lines[cat] += nchanged
    if (cat == "RESIDUE" && nchanged > 0) printf "=== %s  %s ===\n%s\n", file, header, body >> out
    nh = 0; nchanged = 0; body = ""
}
/^=== DIFF /   { flush(); file = $3; next }
/^@@/          { flush(); header = $0; nh = 1; next }
nh == 1        { body = body $0 "\n"; if ($0 ~ /^[+-][^+-]/) nchanged++ }
END { flush()
      printf "STRICT_TWOLEVEL  hunks=%-4d changed_lines=%d\n", hunks["STRICT_TWOLEVEL"], lines["STRICT_TWOLEVEL"]
      printf "RESIDUE          hunks=%-4d changed_lines=%d  -> %s\n", hunks["RESIDUE"], lines["RESIDUE"], out }
' "$IN"
