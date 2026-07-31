#!/usr/bin/env bash
# Tags: no-fasttest

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

set -e

# The right side is read by several threads, so `unified_hash` fills its single hash table concurrently.
# Every query aggregates, because the order in which right blocks are inserted is not deterministic.
COMMON=(--enable_analyzer=1 --max_threads=16 --max_block_size=10000)

# Duplicate keys on the right, for ALL strictness.
RIGHT_DUP="SELECT number % 200000 AS k, toInt64(number) AS v, toString(number % 200000) AS s FROM numbers_mt(300000)"
# Unique keys on the right, for ANY/SEMI/ANTI, where a match is picked arbitrarily among duplicates.
RIGHT_UNIQ="SELECT number AS k, toInt64(number) * 2 AS v, toString(number) AS s FROM numbers_mt(200000)"
LEFT="SELECT number % 250000 AS k, toString(number % 250000) AS s FROM numbers_mt(300000)"

compare_join_algorithms()
{
    local query="$1"
    shift
    local hash_result unified_result

    hash_result=$($CLICKHOUSE_CLIENT "${COMMON[@]}" "$@" --join_algorithm=hash -q "$query")
    unified_result=$($CLICKHOUSE_CLIENT "${COMMON[@]}" "$@" --join_algorithm=unified_hash -q "$query")

    if [ "$hash_result" != "$unified_result" ]; then
        echo "Mismatch for query:"
        echo "$query"
        echo "extra settings: $*"
        echo "--- hash ---"
        echo "$hash_result"
        echo "--- unified_hash ---"
        echo "$unified_result"
        exit 1
    fi
}

run_all_queries()
{
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(l.k), sum(r.v) FROM l INNER JOIN r ON l.k = r.k" "$@"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(l.k), sum(r.v) FROM l LEFT JOIN r ON l.k = r.k" "$@"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(l.k), sum(r.v) FROM l RIGHT JOIN r ON l.k = r.k" "$@"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(l.k), sum(r.v) FROM l FULL JOIN r ON l.k = r.k" "$@"

    # String key, to build a map other than key64.
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(r.v) FROM l INNER JOIN r ON l.s = r.s" "$@"
    # Two key columns.
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(r.v) FROM l INNER JOIN r ON l.k = r.k AND l.s = r.s" "$@"

    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_UNIQ)
        SELECT count(), sum(l.k), sum(r.v) FROM l LEFT ANY JOIN r ON l.k = r.k" "$@"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_UNIQ)
        SELECT count(), sum(l.k) FROM l LEFT SEMI JOIN r ON l.k = r.k" "$@"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_UNIQ)
        SELECT count(), sum(l.k) FROM l LEFT ANTI JOIN r ON l.k = r.k" "$@"
}

# Plain in-memory join.
run_all_queries
# SpillingHashJoin keeping everything in memory.
run_all_queries --max_bytes_before_external_join=100000000
# SpillingHashJoin switching to GraceHashJoin in the middle of a parallel build.
run_all_queries --max_bytes_before_external_join=8000000

# The right side must actually be filled by more than one thread.
count_filling_transforms()
{
    $CLICKHOUSE_CLIENT --enable_analyzer=1 --max_threads=8 --join_algorithm=unified_hash "$@" -q \
        "EXPLAIN PIPELINE WITH l AS ($LEFT), r AS ($RIGHT_DUP)
            SELECT count() FROM l INNER JOIN r ON l.k = r.k" | grep -c FillingRightJoinSide
}

for transforms in $(count_filling_transforms) $(count_filling_transforms --max_bytes_before_external_join=100000000)
do
    if [ "$transforms" -lt 2 ]; then
        echo "Expected the right side to be filled in parallel, got $transforms FillingRightJoinSide"
        exit 1
    fi
done

echo "OK"
