#!/usr/bin/env bash
# U3: normalize away mechanical fork noise (namespace Unified wrapper, include-path
# rewrites, Unified:: qualifications) so the residual diff is real divergence only.
set -u
cd /mnt/ch/ClickHouse

OUT=tmp/uhj_parity/U3_normdiff.txt
: > "$OUT"

norm_uhj() {
    sed -e 's#Interpreters/UnifiedHashJoin/#Interpreters/HashJoin/#g' \
        -e 's#Interpreters/HashJoin/joinDispatch.h#Interpreters/joinDispatch.h#g' \
        -e 's#\bUnified::##g' \
        "$1" \
    | awk '
        # drop the "namespace Unified" opener line and the "{" that follows it
        /^namespace Unified[[:space:]]*$/ { skip_brace = 1; next }
        skip_brace && /^\{[[:space:]]*$/  { skip_brace = 0; next }
        { skip_brace = 0; print }' \
    | awk '
        { lines[NR] = $0 }
        END {
            n = NR
            # remove one trailing blank-line + "}" pair introduced by the wrapper
            while (n > 0 && lines[n] ~ /^[[:space:]]*$/) n--
            if (n > 0 && lines[n] == "}") n--
            while (n > 0 && lines[n] ~ /^[[:space:]]*$/) n--
            for (i = 1; i <= n; i++) print lines[i]
        }'
}

for f in $(ls src/Interpreters/HashJoin/); do
    u="src/Interpreters/UnifiedHashJoin/$f"
    [ -f "$u" ] || { echo "=== BASEONLY $f ===" >> "$OUT"; continue; }
    norm_uhj "$u" > /tmp/u3_norm_uhj.$$
    # baseline gets the same trailing-blank trim so the comparison is fair
    awk '{lines[NR]=$0} END {n=NR; while (n>0 && lines[n] ~ /^[[:space:]]*$/) n--; for(i=1;i<=n;i++) print lines[i]}' \
        "src/Interpreters/HashJoin/$f" > /tmp/u3_norm_base.$$
    if ! diff -q /tmp/u3_norm_base.$$ /tmp/u3_norm_uhj.$$ > /dev/null; then
        echo "=== DIFF $f ===" >> "$OUT"
        diff -u --label "base/$f" --label "uhj/$f" /tmp/u3_norm_base.$$ /tmp/u3_norm_uhj.$$ >> "$OUT"
    fi
done
rm -f /tmp/u3_norm_uhj.$$ /tmp/u3_norm_base.$$

echo "--- residual per-file changed-line counts ---"
awk '/^=== DIFF /{f=$3} /^[+-][^+-]/{c[f]++} END {for (k in c) printf "%6d %s\n", c[k], k}' "$OUT" | sort -rn
echo "TOTAL_RESIDUAL_LINES=$(grep -c '^[+-][^+-]' "$OUT")"
echo "OUT=$OUT"
