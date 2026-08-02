#!/usr/bin/env bash
# Differential check: hash vs unified_hash on the paths touched by the two-level/bucket removal.
# Focus: the rewritten NotJoinedHash scan (RIGHT/FULL non-joined output, including its
# multi-block resume), fillNullsFromBlocks after dropping the bucket guard, the ASOF
# selector-narrowing path, multi-disjunct OR (packed_rank), and LowCardinality keys.

CH="/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse client --host 127.0.0.1 --port 9101 --enable_analyzer=1"
fail=0
n=0

check()
{
    local label="$1"; shift
    local query="$1"; shift
    local h u
    n=$((n + 1))
    h=$($CH "$@" --join_algorithm=hash -q "$query" 2>&1)
    u=$($CH "$@" --join_algorithm=unified_hash -q "$query" 2>&1)
    if [ "$h" != "$u" ]; then
        fail=$((fail + 1))
        echo "MISMATCH [$label] extra=[$*]"
        echo "  query: $query"
        echo "  hash        : $(echo "$h" | head -5 | tr '\n' '|')"
        echo "  unified_hash: $(echo "$u" | head -5 | tr '\n' '|')"
    fi
}

L="SELECT number % 1000 AS k, toString(number % 700) AS s FROM numbers(5000)"
R="SELECT number % 1300 AS k, toInt64(number) AS v, toString(number % 900) AS s FROM numbers(4000)"
# Nullable keys on both sides, so RIGHT/FULL must emit rows whose key is NULL.
LN="SELECT if(number % 7 = 0, NULL, number % 500) AS k FROM numbers(3000)"
RN="SELECT if(number % 5 = 0, NULL, number % 600) AS k, toInt64(number) AS v FROM numbers(3000)"

# --- the rewritten NotJoinedHash scan: every kind/strictness that emits non-joined right rows
for strict in "ALL" "ANY" "SEMI" "ANTI"; do
    for kind in "RIGHT" "FULL"; do
        # FULL SEMI/ANTI are not valid SQL; skip.
        if [ "$kind" = "FULL" ] && { [ "$strict" = "SEMI" ] || [ "$strict" = "ANTI" ]; }; then continue; fi
        check "$kind $strict" "WITH l AS ($L), r AS ($R)
            SELECT count(), sum(ifNull(l.k, -1)), sum(ifNull(r.v, -1)) FROM l $kind $strict JOIN r ON l.k = r.k"
    done
done

# --- multi-block resume of the non-joined stream (the `++it; break;` path I rewrote)
for mbr in 7 64 1000; do
    check "FULL ALL resume max_joined_block_rows=$mbr" "WITH l AS ($L), r AS ($R)
        SELECT count(), sum(ifNull(l.k, -1)), sum(ifNull(r.v, -1)) FROM l FULL JOIN r ON l.k = r.k" \
        --max_joined_block_rows=$mbr
    check "RIGHT ALL resume max_block_size=$mbr" "WITH l AS ($L), r AS ($R)
        SELECT count(), sum(ifNull(r.v, -1)) FROM l RIGHT JOIN r ON l.k = r.k" --max_block_size=$mbr
done

# --- fillNullsFromBlocks: NULL keys must appear exactly once in RIGHT/FULL output
for kind in "RIGHT" "FULL"; do
    check "$kind nullable keys" "WITH l AS ($LN), r AS ($RN)
        SELECT count(), countIf(l.k IS NULL), countIf(r.k IS NULL), sum(ifNull(r.v, -1)) FROM l $kind JOIN r ON l.k = r.k"
    check "$kind nullable keys join_use_nulls" "WITH l AS ($LN), r AS ($RN)
        SELECT count(), countIf(l.k IS NULL), countIf(r.k IS NULL) FROM l $kind JOIN r ON l.k = r.k" --join_use_nulls=1
    check "$kind nullable small blocks" "WITH l AS ($LN), r AS ($RN)
        SELECT count(), countIf(r.k IS NULL) FROM l $kind JOIN r ON l.k = r.k" --max_joined_block_rows=13
done

# --- multi-disjunct OR: exercises mergeJoinMethods/packed_rank, and flag_per_row non-joined output
check "OR two disjuncts INNER" "WITH l AS ($L), r AS ($R)
    SELECT count(), sum(r.v) FROM l INNER JOIN r ON l.k = r.k OR l.s = r.s"
check "OR two disjuncts RIGHT" "WITH l AS ($L), r AS ($R)
    SELECT count(), sum(ifNull(r.v, -1)) FROM l RIGHT JOIN r ON l.k = r.k OR l.s = r.s"
check "OR two disjuncts FULL" "WITH l AS ($L), r AS ($R)
    SELECT count(), sum(ifNull(r.v, -1)) FROM l FULL JOIN r ON l.k = r.k OR l.s = r.s"
check "OR two disjuncts FULL small blocks" "WITH l AS ($L), r AS ($R)
    SELECT count() FROM l FULL JOIN r ON l.k = r.k OR l.s = r.s" --max_joined_block_rows=11
# Mixed-width packed keys across disjuncts -> packed_rank must widen, not fall back.
check "OR mixed packed widths" "WITH l AS (SELECT toUInt16(number % 300) AS a, toUInt32(number % 400) AS b FROM numbers(2000)),
    r AS (SELECT toUInt16(number % 350) AS a, toUInt32(number % 450) AS b FROM numbers(2000))
    SELECT count() FROM l FULL JOIN r ON l.a = r.a OR l.b = r.b"

# --- ASOF, including the nullable-ASOF-key path that narrows the selector
check "ASOF INNER" "WITH l AS (SELECT number % 100 AS k, toInt64(number) AS t FROM numbers(2000)),
    r AS (SELECT number % 100 AS k, toInt64(number * 3) AS t, toInt64(number) AS v FROM numbers(2000))
    SELECT count(), sum(r.v) FROM l ASOF JOIN r ON l.k = r.k AND l.t >= r.t"
check "ASOF LEFT" "WITH l AS (SELECT number % 100 AS k, toInt64(number) AS t FROM numbers(2000)),
    r AS (SELECT number % 100 AS k, toInt64(number * 3) AS t, toInt64(number) AS v FROM numbers(2000))
    SELECT count(), sum(ifNull(r.v, -1)) FROM l ASOF LEFT JOIN r ON l.k = r.k AND l.t >= r.t"
check "ASOF nullable key selector narrowing" "WITH l AS (SELECT number % 100 AS k, toInt64(number) AS t FROM numbers(2000)),
    r AS (SELECT number % 100 AS k, if(number % 4 = 0, NULL, toInt64(number * 3)) AS t, toInt64(number) AS v FROM numbers(2000))
    SELECT count(), sum(ifNull(r.v, -1)) FROM l ASOF LEFT JOIN r ON l.k = r.k AND l.t >= r.t"

# --- LowCardinality single key (the gate whose !use_two_level_maps conjunct I dropped)
LLC="SELECT toLowCardinality(toString(number % 500)) AS k FROM numbers(3000)"
RLC="SELECT toLowCardinality(toString(number % 600)) AS k, toInt64(number) AS v FROM numbers(3000)"
for kind in "INNER" "LEFT" "RIGHT" "FULL"; do
    check "LowCardinality $kind" "WITH l AS ($LLC), r AS ($RLC)
        SELECT count(), sum(ifNull(r.v, -1)) FROM l $kind JOIN r ON l.k = r.k"
done
check "LowCardinality Nullable" "WITH l AS (SELECT toLowCardinality(if(number % 9 = 0, NULL, toString(number % 300))) AS k FROM numbers(2000)),
    r AS (SELECT toLowCardinality(if(number % 6 = 0, NULL, toString(number % 400))) AS k, toInt64(number) AS v FROM numbers(2000))
    SELECT count(), sum(ifNull(r.v, -1)) FROM l FULL JOIN r ON l.k = r.k"

# --- spilling / grace paths, which drive a second in-memory instance
for extra in "--max_bytes_before_external_join=50000000" "--max_bytes_before_external_join=2000000"; do
    for kind in "INNER" "RIGHT" "FULL"; do
        check "spill $kind" "WITH l AS ($L), r AS ($R)
            SELECT count(), sum(ifNull(r.v, -1)) FROM l $kind JOIN r ON l.k = r.k" $extra
    done
done

# --- parallel build with many threads, all kinds
for kind in "INNER" "LEFT" "RIGHT" "FULL"; do
    check "parallel $kind" "WITH l AS ($L), r AS ($R)
        SELECT count(), sum(ifNull(r.v, -1)) FROM l $kind JOIN r ON l.k = r.k" --max_threads=16 --max_block_size=100
done

# --- empty right side and empty left side
check "empty right FULL" "WITH l AS ($L), r AS (SELECT number AS k, toInt64(number) AS v FROM numbers(0))
    SELECT count(), sum(ifNull(r.v, -1)) FROM l FULL JOIN r ON l.k = r.k"
check "empty left FULL" "WITH l AS (SELECT number AS k FROM numbers(0)), r AS ($R)
    SELECT count(), sum(ifNull(r.v, -1)) FROM l FULL JOIN r ON l.k = r.k"
check "empty left RIGHT" "WITH l AS (SELECT number AS k FROM numbers(0)), r AS ($R)
    SELECT count(), sum(ifNull(r.v, -1)) FROM l RIGHT JOIN r ON l.k = r.k"

echo "checks=$n mismatches=$fail"
[ "$fail" -eq 0 ] && echo "DIFF_RESULT=PASS" || echo "DIFF_RESULT=FAIL"
