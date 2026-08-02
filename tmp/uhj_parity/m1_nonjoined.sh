#!/usr/bin/env bash
# M1 refute test: the non-joined rows of RIGHT/FULL joins must be byte-identical between
# `hash` and `unified_hash`. A wrong bucket partition shows up here as duplicated or lost rows,
# which aggregates alone could mask, so this compares the full sorted row set as well.
set -uo pipefail
CH=(/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse client --host 127.0.0.1 --port "${UHJ_PORT:-9101}")

RIGHT="SELECT number % 200000 AS k, toInt64(number) AS v FROM numbers_mt(300000)"
# Left covers only part of the right key space, so a large set of right rows stays non-joined.
LEFT="SELECT number % 60000 AS k FROM numbers_mt(100000)"
# A right side with NULL keys, to exercise the stream-0 nullmap guard.
RIGHT_NULL="SELECT if(number % 7 = 0, NULL, number % 50000) AS k, toInt64(number) AS v FROM numbers_mt(120000)"

fail=0
cmp_case()
{
    local label="$1" query="$2"; shift 2
    local a b
    a=$("${CH[@]}" --enable_analyzer=1 --join_algorithm=hash "$@" -q "$query")
    b=$("${CH[@]}" --enable_analyzer=1 --join_algorithm=unified_hash "$@" -q "$query")
    if [ "$a" != "$b" ]; then
        echo "MISMATCH [$label] extra=$*"
        diff <(echo "$a") <(echo "$b") | head -20
        fail=1
    else
        echo "ok [$label] extra=$*"
    fi
}

for threads in 1 4 16; do
    for kind in RIGHT FULL; do
        # Aggregate form: totals over the whole result.
        cmp_case "$kind agg" "WITH l AS ($LEFT), r AS ($RIGHT)
            SELECT count(), sum(r.v), sum(ifNull(l.k, -1)) FROM l $kind JOIN r ON l.k = r.k" \
            --max_threads="$threads"
        # Full row-set form: catches duplication/loss that sums could cancel out.
        cmp_case "$kind rowset" "WITH l AS ($LEFT), r AS ($RIGHT)
            SELECT r.k, r.v, ifNull(l.k, -1) AS lk FROM l $kind JOIN r ON l.k = r.k
            ORDER BY r.k, r.v, lk" \
            --max_threads="$threads"
        # NULL keys on the right: the nullmap path must be emitted exactly once.
        cmp_case "$kind nulls" "WITH l AS ($LEFT), r AS ($RIGHT_NULL)
            SELECT count(), sum(r.v), countIf(r.k IS NULL) FROM l $kind JOIN r ON l.k = r.k" \
            --max_threads="$threads"
        # String keys route through a different map family than the numeric ones.
        cmp_case "$kind string" "WITH l AS (SELECT toString(number % 60000) AS s FROM numbers_mt(100000)),
                                      r AS (SELECT toString(number % 200000) AS s, toInt64(number) AS v FROM numbers_mt(300000))
            SELECT count(), sum(r.v) FROM l $kind JOIN r ON l.s = r.s" \
            --max_threads="$threads"
        # Small UInt8 key: a direct-addressed map, where iteration is a single bucket.
        cmp_case "$kind fixed8" "WITH l AS (SELECT toUInt8(number % 50) AS k FROM numbers_mt(5000)),
                                      r AS (SELECT toUInt8(number % 200) AS k, toInt64(number) AS v FROM numbers_mt(20000))
            SELECT count(), sum(r.v) FROM l $kind JOIN r ON l.k = r.k" \
            --max_threads="$threads"
    done
done

# The join must actually advertise the parallel non-joined capability, and the pipeline must show
# more than one non-joined stream; otherwise the partitioning code above is never exercised.
streams=$("${CH[@]}" --enable_analyzer=1 --max_threads=8 --join_algorithm=unified_hash -q \
    "EXPLAIN PIPELINE WITH l AS ($LEFT), r AS ($RIGHT)
        SELECT count() FROM l RIGHT JOIN r ON l.k = r.k" | grep -c 'FillingRightJoinSide\|JoiningTransform')
echo "PIPELINE_TRANSFORMS=$streams"

echo "FAIL=$fail"
[ "$fail" -eq 0 ] && echo "M1 ROWSETS IDENTICAL — refute condition not met"
echo "JOB_EXIT=$fail"
exit "$fail"
